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
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Handle additional arguments separately if necessary
    additional_args = parser.parse_known_args()[1]  # This gets the remaining arguments

    # Check if there are any remaining arguments and assign them properly
    if additional_args:
        model_args.architecture_type = additional_args[0]
    
    # Pass the arguments to the experiment setup
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()

if __name__ == "__main__":
    main()
