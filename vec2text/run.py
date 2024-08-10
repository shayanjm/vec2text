import transformers
from experiments import experiment_from_args
from run_args import DataArguments, ModelArguments, TrainingArguments


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    # Add the model_type argument manually
    parser.add_argument("--model_type", choices=["mlp", "attention"], required=True, help="Model type to use (mlp or attention)")
    
    # Parse arguments including the new model_type argument
    model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # Assign the model_type to the model_args
    model_args.model_type = additional_args[0]  # Assuming model_type is the only additional argument
    
    # Pass the arguments to the experiment setup
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()

if __name__ == "__main__":
    main()
