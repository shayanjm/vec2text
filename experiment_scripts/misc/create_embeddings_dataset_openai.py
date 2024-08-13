import argparse
import functools
import math
from typing import Dict

import datasets
import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed

from vec2text.data_helpers import (
    load_beir_datasets,
    load_standard_val_datasets,
    retain_dataset_columns,
)
from vec2text.models import load_embedder_and_tokenizer
from vec2text.models.model_utils import mean_pool
from vec2text.utils import dataset_map_multi_worker

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process dataset and embedders")
    parser.add_argument("dataset", type=str, help="Path or name of the dataset")
    parser.add_argument("--max_tokens",
                        type=int,
                        default=128,
                        help="Maximum number of tokens for truncation (default: 128)")
    return parser.parse_args()


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def embed_openai(example: Dict, model_name: str, max_tokens: int) -> Dict:
    from concurrent.futures import ThreadPoolExecutor

    encoding = tiktoken.encoding_for_model(model_name)

    # Encode and truncate text as needed
    text_tokens = encoding.encode_batch(example["text"])
    text_tokens = [tok[:max_tokens] for tok in text_tokens]
    text_list = encoding.decode_batch(text_tokens)

    # Initialize OpenAI Client
    client = openai.OpenAI()  # Ensure correct client initialization

    batches = math.ceil(len(text_list) / 128)
    outputs = []

    for i in range(len(text_list)):
        if len(text_list[i]) == 0:
            print(f"warning: set element {i} to a random sequence")
            text_list[i] = "random sequence"

    def process_batch(batch):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch, model=model_name, encoding_format="float"
        )
        return [e.embedding for e in response.data]

    with ThreadPoolExecutor() as executor:
        batch_indices = range(batches)
        results = executor.map(process_batch, batch_indices)

        for result in results:
            outputs.extend(result)

    example["text"] = text_list
    example["embeddings_A"] = outputs

    return example


def main():
    datasets.disable_caching()
    args = parse_args()
    model_name = "text-embedding-3-large"
    
    full_name = "__".join(
        (
            args.dataset,
            f"trunc-{args.max_tokens}"
            model_name,
        )
    )
    print(f"Embedding {args.dataset} dataset using {model_name}. Truncating to {args.max_tokens} tokens. Saving dataset to HuggingFace as {full_name}")
    all_datasets = {
        **load_standard_val_datasets(),
        **load_beir_datasets(),
    }
    print("Available datasets:", all_datasets.keys())
    assert (
        args.dataset in all_datasets
    ), f"unknown dataset {args.dataset}; choices {all_datasets.keys()}"
    dataset = all_datasets[args.dataset]
    max_tokens = args.max_tokens

    print(f"[*] embedding {args.dataset}")
    # Use functools.partial to pass the model_name to embed_openai
    map_fn = functools.partial(embed_openai, model_name=model_name, max_tokens=max_tokens)

    dataset = dataset_map_multi_worker(
        dataset,
        batched=True,
        batch_size=128,
        map_fn=map_fn,  # Use the partial function here
        num_proc=1,
    )
    print(f"pushing to hub with name {full_name}")

    dataset = retain_dataset_columns(dataset, ["text", "embeddings_A"])
    dataset.push_to_hub(full_name)
    print("done")


if __name__ == "__main__":
    main()
