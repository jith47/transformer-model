# train_chatbot.py
from datasets import load_dataset
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import torch
from evaluate import load

# 1. Initialize Model & Tokenizer ==============================================
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# 2. Configure LoRA Adapters ==================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],  # Added k_proj
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    modules_to_save=["embed_tokens", "embed_positions"]  # Critical fix
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Dataset Preparation ======================================================
def format_chat_samples(examples):
    inputs = []
    targets = []
    for persona, history in zip(examples["personality"], examples["history"]):
        # Ensure valid conversation pairs
        if len(history) < 2:
            continue
            
        # Format with explicit speaker tags
        context = "\n".join([
            f"<USER> {utt.strip()}" if i%2 == 0 else f"<BOT> {utt.strip()}"
            for i, utt in enumerate(history[:-1])
        ])
        inputs.append(f"Persona: {'; '.join(persona)}\n{context}")
        targets.append(history[-1].strip())
    
    return {"input": inputs, "target": targets}

def tokenize_dataset(examples):
    tokenized_inputs = tokenizer(
        examples["input"],
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized_targets = tokenizer(
        examples["target"],
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Explicit index validation
    max_valid_id = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder) - 1
    assert tokenized_inputs["input_ids"].max() <= max_valid_id, "Input ID overflow"
    assert tokenized_targets["input_ids"].max() <= max_valid_id, "Label ID overflow"
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"]
    }

# 4. Load Dataset with Remote Code Trust ======================================
dataset = load_dataset(
    "bavard/personachat_truecased",
    trust_remote_code=True  # Required for this dataset
)

# Process dataset
dataset = dataset.map(format_chat_samples, batched=True, remove_columns=dataset["train"].column_names)
dataset = dataset.filter(lambda x: len(x["input"]) > 0)  # Remove empty samples
tokenized_dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=["input", "target"])

# 5. Training Setup ===========================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir="./chatbot_model",
    num_train_epochs=2,  # Reduced for testing
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=False,  # Disable FP16 for stability
    optim="adamw_torch",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    report_to="none",
    remove_unused_columns=True,
    label_names=["labels"]
)

# 6. Trainer Configuration ===================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Start Training ==========================================================
print("Starting training...")
trainer.train()
model.save_pretrained("./trained_chatbot")