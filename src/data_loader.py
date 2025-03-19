from datasets import load_dataset

def load_daily_dialog():
    """Load and preprocess DailyDialog dataset."""
    dataset = load_dataset("daily_dialog")
    
    def process_examples(examples):
        inputs, targets = [], []
        for conv in examples["dialog"]:
            # Split conversation into input-response pairs
            utterances = [u.strip() for u in conv if u.strip()]
            for i in range(len(utterances) - 1):
                inputs.append(utterances[i])
                targets.append(utterances[i+1])
        return {"input": inputs, "target": targets}
    
    dataset = dataset.map(
        process_examples,
        batched=True,
        remove_columns=["dialog", "act", "emotion"],
    )
    return dataset