
__all__ = [
    "train_with_existing_labels",
]

def train_with_existing_labels(trainer):
    """
    Trains using existing labels from JSONL format data where each line contains
    input_ids, attention_mask, and labels.
    
    Parameters:
    trainer: The trainer object containing the datasets
    
    Returns:
    trainer: The modified trainer object with properly formatted datasets
    """
    def _process_existing_labels(examples):
        """
        Process the examples maintaining their existing labels.
        
        Parameters:
        examples: Dictionary containing 'input_ids', 'attention_mask', and 'labels'
        
        Returns:
        Dictionary with processed labels
        """
        # For JSONL format, labels are already present
        # Just ensure they're in the correct format
        if "labels" not in examples:
            raise ValueError("Labels not found in the dataset. Ensure your JSONL contains 'labels' field.")
        
        # Convert labels to list if they're not already
        if isinstance(examples["labels"], (list, tuple)):
            return {"labels": examples["labels"]}
        else:
            return {"labels": [examples["labels"]]}

    # Process training dataset if it exists
    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        trainer.train_dataset = trainer.train_dataset.map(
            _process_existing_labels,
            batched=True,
            desc="Processing training dataset"
        )

    # Process evaluation dataset if it exists
    if hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
        # Handle both dictionary and single dataset cases
        if isinstance(trainer.eval_dataset, dict):
            for key, value in trainer.eval_dataset.items():
                trainer.eval_dataset[key] = value.map(
                    _process_existing_labels,
                    batched=True,
                    desc=f"Processing evaluation dataset {key}"
                )
        else:
            trainer.eval_dataset = trainer.eval_dataset.map(
                _process_existing_labels,
                batched=True,
                desc="Processing evaluation dataset"
            )

    return trainer