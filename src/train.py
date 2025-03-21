from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU Score
    bleu_result = bleu.compute(
        predictions=decoded_preds, 
        references=[[ref] for ref in decoded_labels]
    )
    
    # ROUGE Score
    rouge_result = rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels
    )
    
    return {
        "bleu": bleu_result["bleu"],
        "rougeL": rouge_result["rougeL"],
    }

# Load model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Add special tokens safely
new_tokens = ["User:", "Bot:"]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
model.resize_token_embeddings(len(tokenizer))

# Tokenize dataset
def tokenize_dataset(examples):
    contexts = []
    targets = []
    
    for personality, history, candidates in zip(examples["personality"], 
                                              examples["history"],
                                              examples["candidates"]):
        # Combine personality with conversation history
        persona_str = " ".join(personality)
        context = f"Persona: {persona_str}\nHistory: {' '.join(history)}"
        
        # Use first candidate as the target response (ground truth)
        target = candidates[0]  # First candidate is the correct response
        
        contexts.append(f"User: {context}")
        targets.append(f"Bot: {target}")

    # Tokenization
    tokenized_inputs = tokenizer(
        contexts,
        max_length=256,
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

    # In your tokenization function
    print(f"Input IDs type: {tokenized_inputs['input_ids'].dtype}")
    print(f"Labels type: {tokenized_targets['input_ids'].dtype}")

    return {
        "input_ids": tokenized_inputs["input_ids"].to(torch.int64),
        "attention_mask": tokenized_inputs["attention_mask"].to(torch.int64),
        "labels": tokenized_targets["input_ids"].to(torch.int64),
    }

# Load dataset
dataset = load_dataset("bavard/personachat_truecased", streaming=True)

# Tokenize with proper column removal
tokenized_dataset = dataset.map(
    lambda x: {
        "input_ids": x["input_ids"],
        "attention_mask": x["attention_mask"],
        "labels": x["labels"]
    },
    batched=True,
    remove_columns=dataset["train"].column_names
)
# Process in smaller batches
tokenized_dataset = dataset.map(
    tokenize_dataset,
    batched=True,
    batch_size=500,  # Reduce if memory constrained
    remove_columns=dataset["train"].column_names
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    max_steps=10000,  # Total training steps (adjust based on needs)
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_steps=1000,
    weight_decay=0.01,
    fp16=True,
    logging_steps=500,
    save_strategy="steps",  # Save every X steps
    save_steps=1000,  # Save checkpoint every 1000 steps
    evaluation_strategy="steps",  # Evaluate every X steps
    eval_steps=1000,  # Evaluate every 1000 steps
    predict_with_generate=True,
    generation_max_length=128,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("transformer-modelmodels/trained/t5_chatbot")