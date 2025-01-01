from vec2text.analyze_utils import load_experiment_and_trainer
from accelerate import Accelerator
import torch
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math

# Ensure distributed training is initialized
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

# Get local rank and set device
local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# Load experiment and trainer
experiment, trainer = load_experiment_and_trainer('./output')

# Initialize Accelerator and associate it with the trainer
accelerator = Accelerator()
trainer.accelerator = accelerator
trainer.args.distributed_state = accelerator.state

# Evaluate the model
if dist.is_initialized():
    dist.barrier()  # Ensure all processes synchronize before evaluation

try:
    print(f"[Rank {local_rank}] Starting evaluation...")

    ############################################
    # 3) Guided Decoding (Single Pass)
    ############################################
    print(f"\n[Rank {local_rank}] === 3) Guided Decoding (Single Pass) ===")
    trainer.gen_kwargs["guided"] = True
    trainer.gen_kwargs["checkpoint_interval"] = 10
    trainer.gen_kwargs["beam_size"] = 5
    trainer.gen_kwargs["multipass"] = 1
    trainer.gen_kwargs["alpha"] = 0.5
    trainer.gen_kwargs["beta"] = 1.5
    trainer.gen_kwargs["length_penalty"] = 1.0

    # Typically set HF generation to beam-style:
    trainer.gen_kwargs["do_sample"] = False
    trainer.gen_kwargs["num_beams"] = 5
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Guided Decoding (Single Pass) metrics]", metrics)
    ############################################
    # BASELINE
    ############################################
    print(f"\n[Rank {local_rank}] === BASELINE ===")
    # If you have any default gen_kwargs you want to reset, do so here:
    trainer.gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": False,
        "num_beams": 1,  # or 5, depending on your baseline
        "no_repeat_ngram_size": 0,
    }
    baseline_metrics = trainer.evaluate()
    print("[BASELINE metrics]", baseline_metrics)

    ############################################
    # 1) Single-Pass, Basic Beam Search
    ############################################
    print(f"\n[Rank {local_rank}] === 1) Single-Pass, Basic Beam Search ===")
    trainer.gen_kwargs["guided"] = False
    trainer.gen_kwargs["beam_size"] = 5       # For your custom logic
    trainer.gen_kwargs["multipass"] = 1
    trainer.gen_kwargs["checkpoint_interval"] = 10  # Not actually used if guided=False, but safe to set
    trainer.gen_kwargs["alpha"] = 1.0         # Not used if guided=False
    trainer.gen_kwargs["beta"] = 1.0          # Not used if guided=False
    trainer.gen_kwargs["length_penalty"] = 1.0

    # Also ensure HF generation-based beam settings
    trainer.gen_kwargs["do_sample"] = False
    trainer.gen_kwargs["num_beams"] = 5
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Single-Pass, Basic Beam Search metrics]", metrics)

    ############################################
    # 2) Single-Pass, Top-k Sampling
    ############################################
    print(f"\n[Rank {local_rank}] === 2) Single-Pass, Top-k Sampling ===")
    trainer.gen_kwargs["guided"] = False
    trainer.gen_kwargs["beam_size"] = 1       # Single path if guided=False
    trainer.gen_kwargs["multipass"] = 1
    trainer.gen_kwargs["alpha"] = 1.0
    trainer.gen_kwargs["beta"] = 1.0
    trainer.gen_kwargs["length_penalty"] = 1.0

    # HF sampling setup
    trainer.gen_kwargs["do_sample"] = True
    trainer.gen_kwargs["top_k"] = 50
    trainer.gen_kwargs["temperature"] = 1.0
    trainer.gen_kwargs["num_beams"] = 1
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Single-Pass, Top-k Sampling metrics]", metrics)

    ############################################
    # 4) Guided Decoding + Higher Checkpoint Frequency
    ############################################
    print(f"\n[Rank {local_rank}] === 4) Guided Decoding, freq=5 ===")
    trainer.gen_kwargs["guided"] = True
    trainer.gen_kwargs["checkpoint_interval"] = 5
    trainer.gen_kwargs["beam_size"] = 3
    trainer.gen_kwargs["multipass"] = 1
    trainer.gen_kwargs["alpha"] = 1.0
    trainer.gen_kwargs["beta"] = 1.0
    trainer.gen_kwargs["length_penalty"] = 1.0

    trainer.gen_kwargs["do_sample"] = False
    trainer.gen_kwargs["num_beams"] = 3
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Guided Decoding, freq=5 metrics]", metrics)

    ############################################
    # 5) Multipass Iterative Refinement
    ############################################
    print(f"\n[Rank {local_rank}] === 5) Multipass Iterative Refinement ===")
    trainer.gen_kwargs["guided"] = True
    trainer.gen_kwargs["checkpoint_interval"] = 10
    trainer.gen_kwargs["beam_size"] = 5
    trainer.gen_kwargs["multipass"] = 2   # or 3
    trainer.gen_kwargs["alpha"] = 1.0
    trainer.gen_kwargs["beta"] = 1.0
    trainer.gen_kwargs["length_penalty"] = 1.2

    trainer.gen_kwargs["do_sample"] = False
    trainer.gen_kwargs["num_beams"] = 5
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Multipass Iterative Refinement metrics]", metrics)

    ############################################
    # 6) Multipass with Strong Embedding Guidance
    ############################################
    print(f"\n[Rank {local_rank}] === 6) Multipass w/ Strong Embedding Guidance ===")
    trainer.gen_kwargs["guided"] = True
    trainer.gen_kwargs["checkpoint_interval"] = 10
    trainer.gen_kwargs["beam_size"] = 5
    trainer.gen_kwargs["multipass"] = 2
    trainer.gen_kwargs["alpha"] = 0.7
    trainer.gen_kwargs["beta"] = 2.0
    trainer.gen_kwargs["length_penalty"] = 1.0

    trainer.gen_kwargs["do_sample"] = False
    trainer.gen_kwargs["num_beams"] = 5
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Multipass (strong embed guidance) metrics]", metrics)

    ############################################
    # 7) Short Checkpoints & Multiple Passes
    ############################################
    print(f"\n[Rank {local_rank}] === 7) Short Checkpoints & Multipass ===")
    trainer.gen_kwargs["guided"] = True
    trainer.gen_kwargs["checkpoint_interval"] = 5
    trainer.gen_kwargs["beam_size"] = 3
    trainer.gen_kwargs["multipass"] = 2
    trainer.gen_kwargs["alpha"] = 1.0
    trainer.gen_kwargs["beta"] = 1.5
    trainer.gen_kwargs["length_penalty"] = 1.0

    trainer.gen_kwargs["do_sample"] = False
    trainer.gen_kwargs["num_beams"] = 3
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Short Checkpoints & Multipass metrics]", metrics)

    ############################################
    # 8) Looser Generation w/ Temperature + Guidance
    ############################################
    print(f"\n[Rank {local_rank}] === 8) Looser Generation + Guidance ===")
    trainer.gen_kwargs["guided"] = True
    trainer.gen_kwargs["checkpoint_interval"] = 10
    trainer.gen_kwargs["beam_size"] = 3
    trainer.gen_kwargs["multipass"] = 1  # or 2
    trainer.gen_kwargs["alpha"] = 1.0
    trainer.gen_kwargs["beta"] = 1.0
    trainer.gen_kwargs["length_penalty"] = 1.0

    trainer.gen_kwargs["do_sample"] = True
    trainer.gen_kwargs["top_k"] = 50
    trainer.gen_kwargs["temperature"] = 1.2
    trainer.gen_kwargs["num_beams"] = 3
    trainer.gen_kwargs["max_new_tokens"] = 64

    metrics = trainer.evaluate()
    print("[Looser Generation w/ Guidance metrics]", metrics)

except Exception as e:
    print(f"[Rank {local_rank}] An error occurred during evaluation: {str(e)}")

# Finalize distributed processing
dist.destroy_process_group()
