from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_dataset(examples):
    """Tokenize inputs and targets."""
    inputs = [f"User: {text}" for text in examples["input"]]
    targets = [f"Bot: {text}" for text in examples["target"]]
    
    # Tokenize inputs and targets
    tokenized_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        targets,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }