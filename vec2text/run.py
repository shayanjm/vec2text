import transformers
from experiments import experiment_from_args
from run_args import DataArguments, ModelArguments, TrainingArguments


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    # Add the architecture_type argument manually
    parser.add_argument("--architecture_type", choices=["mlp", "attention"], required=True, help="Architecture type to use (mlp or attention)")
    
    # Parse arguments including the new architecture_type argument
    model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    print(f"Additional Args: {additional_args}")
    
    # Assign the architecture type to the model_args
    model_args.architecture_type = additional_args[0]  # Assuming architecture_type is the only additional argument
    
    # Pass the arguments to the experiment setup
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()

if __name__ == "__main__":
    main()
