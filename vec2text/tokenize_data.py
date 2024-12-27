from typing import Callable, Dict

import torch
import transformers

from vec2text.models import InversionModel


def tokenize_function(
    tokenizer: transformers.PreTrainedTokenizer,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    text_column_name: str,
    max_seq_length: int,
    padding: bool = False,
    prefix: str = None,
) -> Callable[[Dict], Dict]:
    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if prefix:
            texts = [f"{prefix}: {text}" for text in examples[text_column_name]]
        else:
            texts = examples[text_column_name]
        output = tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def tokenize_function_llama_chat(
    tokenizer,
    embedder_tokenizer,
    text_column_name,
    max_seq_length,
    padding: bool = False,
    # no-op for compatibility with other tokenization functions
    prefix: str = None,
) -> Callable[[Dict], Dict]:
    """Use special tokenization for LLAMA chat models."""

    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if "prefix" not in examples:
            # hacky way to turn datasets into the right format for LLAMA chat.
            # "real" prompt datasets like one_million_paired_instructions
            # have "prefix" and "suffix" already.
            #
            # so this is only for evaluation datasets that may not have
            # actual prefix-suffix pairing.
            #
            examples["prefix"] = [""] * len(examples[text_column_name])
            examples["suffix"] = examples[text_column_name]

        output = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            text=[
                f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
                for (system_message, instruction) in zip(
                    examples["prefix"], examples["suffix"]
                )
            ],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def embed_dataset_batch(model, batch: Dict) -> Dict:
    """
    Pre-compute frozen embeddings for each example in `batch`.
    If `model.config.embedding_transform_strategy == "overlap_chunking"`, 
    produce shape (num_chunks, embed_dim) for each example, then 
    PAD/TRUNCATE to (max_num_chunks, embed_dim). 
    Otherwise, produce shape (embed_dim,) as before.
    """

    # Basic checks
    assert "input_ids" in batch, f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    # The raw input_ids are from model.tokenizer. We'll decode them to get strings:
    input_ids = batch["input_ids"]  # shape: (batch_size, seq_len)
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    # We'll store the final embeddings in this list.
    # Each item will be a numpy array of shape (embed_dim,) [old strategy]
    # or (max_num_chunks, embed_dim) [overlap_chunking with pad/trunc].
    all_frozen_embs = []

    # For chunking config:
    embedding_strategy = model.config.embedding_transform_strategy
    chunk_size = getattr(model.config, "chunk_size", 128)
    chunk_overlap = getattr(model.config, "chunk_overlap", 16)

    # -- NEW: define how many chunks we want to keep per example --
    # If your dataset can handle up to 10 chunks, for example:
    max_num_chunks = getattr(model.config, "max_num_chunks", 10)

    for text_str in inputs_str:
        # 1) Convert text -> tokens for the embedder.
        emb_input = model.embedder_tokenizer(
            text_str,
            truncation=False,    # let us see the entire sequence so we can chunk fully
            padding=False,
            return_tensors="pt",
        ).to(next(model.parameters()).device)

        if embedding_strategy == "overlap_chunking":
            # Extract the tokenized IDs for chunking
            token_ids = emb_input["input_ids"].squeeze(0)       # shape: (seq_len,)
            token_mask = emb_input["attention_mask"].squeeze(0) # shape: (seq_len,)

            chunk_embs = []
            start = 0
            while start < token_ids.size(0):
                end = min(start + chunk_size, token_ids.size(0))
                chunk_ids = token_ids[start:end]
                chunk_attention = token_mask[start:end]

                # Wrap them into batch form for call_embedding_model
                chunk_ids = chunk_ids.unsqueeze(0)      # (1, chunk_len)
                chunk_attention = chunk_attention.unsqueeze(0)

                with torch.no_grad():
                    # call_embedding_model => (1, embed_dim)
                    chunk_embedding = model.call_embedding_model(
                        input_ids=chunk_ids,
                        attention_mask=chunk_attention,
                    )
                # => shape: (embed_dim,)
                chunk_embs.append(chunk_embedding.squeeze(0))

                step_size = chunk_size - chunk_overlap
                if step_size <= 0:
                    raise ValueError("chunk_overlap must be smaller than chunk_size.")
                start += step_size

            if len(chunk_embs) == 0:
                # Edge case of empty input
                chunk_embs = [torch.zeros(model.embedder_dim, device="cpu")]

            # Stack => (num_chunks, embed_dim)
            chunk_embs_stacked = torch.stack(chunk_embs, dim=0)

            # --------- PAD OR TRUNCATE TO (max_num_chunks, embed_dim) -----------
            num_chunks = chunk_embs_stacked.size(0)
            if num_chunks < max_num_chunks:
                # Pad
                padded_tensor = torch.zeros(
                    (max_num_chunks, model.embedder_dim),
                    device=chunk_embs_stacked.device,
                )
                padded_tensor[:num_chunks] = chunk_embs_stacked
                chunk_embs_stacked = padded_tensor
            elif num_chunks > max_num_chunks:
                # Truncate
                chunk_embs_stacked = chunk_embs_stacked[:max_num_chunks]

            # Now shape => (max_num_chunks, embed_dim)
            all_frozen_embs.append(chunk_embs_stacked.cpu().numpy())

        else:
            # Fallback: the old single pooled embedding approach (shape => (1, embed_dim))
            with torch.no_grad():
                single_emb = model.call_embedding_model(**emb_input)
            # store shape => (embed_dim,)
            all_frozen_embs.append(single_emb.squeeze(0).cpu().numpy())

    # Now we have a list of embeddings.
    # With chunking: shape => (max_num_chunks, embed_dim) for each example
    # Without chunking: shape => (embed_dim,) for each example
    batch["frozen_embeddings"] = all_frozen_embs
    return batch

def get_tokenizer_mapping(
    lm: str, inverter: str, inverter_vocab_size: int
) -> torch.Tensor:
    """Computes the mapping from token outputs in `lm`'s vocabulary to those in `inverter's
    vocabulary. Makes some assumptions about spacing.
    """
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(lm)
    inverter_tokenizer = transformers.AutoTokenizer.from_pretrained(inverter)

    lm_vocab = lm_tokenizer.vocab
    mapping = torch.zeros(len(lm_vocab), dtype=torch.long)
    for k, idx in lm_tokenizer.vocab.items():
        # We replace space tokens with nothing and allow the call to
        # inverter_tokenizer.decode to determine this. We also
        # filter out 2 and 3 as first tokens which are extremely common
        # when the T5 tokenizer processes unicode. (These are hacks
        # specific to the LLAMA-T5 lm-inverter pairing, and it would
        # be better to find an automated wa to do this later.)
        mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[0]
        if mapping[idx] in [2, 3]:
            mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[1]

    preservation = len(set(mapping.tolist())) / len(lm_vocab)
    print(
        f"Mapped tokenizer {lm} to {inverter}. Preserved {preservation*100:.1f}% of unique tokens."
    )
    return mapping
