# train_chatbot.py
from datasets import load_dataset
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch
from evaluate import load
import bitsandbytes as bnb

# 1. Initialize Model & Tokenizer ==============================================
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# 2. Configure LoRA Adapters ==================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load & Prepare Dataset ===================================================
def format_chat_samples(examples):
    formatted = []
    for persona, history in zip(examples["personality"], examples["history"]):
        # Build conversation with special tokens
        dialog = []
        for i, utt in enumerate(history):
            speaker = "<USER>" if i % 2 == 0 else "<BOT>"
            dialog.append(f"{speaker} {utt.strip()}")
        
        # Combine persona and dialog
        formatted_text = (
            f"Persona: {'; '.join(persona)}"
            f"Conversation:{dialog}<BOT>"
        )
        formatted.append(formatted_text)
    
    return {"text": formatted}

def tokenize_dataset(examples):
    tokenized = tokenizer(
        examples["text"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()
    }

dataset = load_dataset("bavard/personachat_truecased")
dataset = dataset.map(format_chat_samples, batched=True)
tokenized_dataset = dataset.map(tokenize_dataset, batched=True)

# 4. Training Setup ===========================================================
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    report_to="none",
    remove_unused_columns=False
)

# 5. Metrics & Evaluation =====================================================
bleu = load("bleu")
rouge = load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    return {
        "bleu": bleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])["bleu"],
        "rougeL": rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
    }

# 6. Initialize Trainer ========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# 7. Start Training ============================================================
print("Starting training...")
trainer.train()
model.save_pretrained("trained/t5_chatbot")

# 8. Inference Function ========================================================
def chat(prompt, persona="helpful assistant", max_length=200):
    formatted_prompt = f"Persona: {persona}\n<USER> {prompt}\n<BOT>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
print(chat("What's your opinion on AI ethics?"))