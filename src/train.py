from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from evaluate import load

# Load BLEU metric
metric = load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}
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
            inputs.append(utterances[i])  # User input
            targets.append(utterances[i+1])  # Bot response
    
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
    num_train_epochs=5,  # Train longer
    per_device_train_batch_size=32,  # Adjust based on GPU memory
    per_device_eval_batch_size=32,
    learning_rate=5e-5,  # Lower learning rate for fine-tuning
    warmup_steps=500,  # Warmup for stable training
    weight_decay=0.01,  # Regularization
    fp16=True,  # Mixed precision for faster training
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    predict_with_generate=True,  # For sequence-to-sequence
    save_total_limit=2,  # Limit checkpoints
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Add evaluation metrics
)

# Train
trainer.train()
model.save_pretrained("transformer-modelmodels/trained/t5_chatbot")