from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Tokenize dataset
def tokenize_dataset(examples):
    inputs, targets = [], []
    for conv in examples["dialog"]:
        # Split conversation into input-response pairs
        utterances = [u.strip() for u in conv if u.strip()]
        for i in range(len(utterances) - 1):
            inputs.append(f"respond: {utterances[i]}")  # Prefix for T5
            targets.append(utterances[i+1])
    
    # Tokenize inputs and targets
    tokenized_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }

# Load dataset
dataset = load_dataset("daily_dialog")

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_dataset,
    batched=True,
    batch_size=1000,  # Adjust based on memory
    remove_columns=["dialog", "act", "emotion"],  # Remove unused columns
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,  # Adjust based on GPU memory
    per_device_eval_batch_size=32,
    fp16=True,
    gradient_accumulation_steps=2,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    predict_with_generate=True,  # For sequence-to-sequence
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

# Train
trainer.train()
model.save_pretrained("models/trained/t5_chatbot")