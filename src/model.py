from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-base"  # More capacity than t5-small
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Add special tokens if needed
tokenizer.add_special_tokens({"additional_special_tokens": ["User:", "Bot:"]})
model.resize_token_embeddings(len(tokenizer))
# Add padding token if missing
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# def tokenize_dataset(examples):
#     """Tokenize inputs and targets."""
#     inputs = [f"User: {text}" for text in examples["input"]]
#     targets = [f"Bot: {text}" for text in examples["target"]]
    
#     # Tokenize inputs and targets
#     tokenized_inputs = tokenizer(
#         inputs,
#         truncation=True,
#         padding="max_length",
#         max_length=128,
#         return_tensors="pt",
#     )
#     tokenized_targets = tokenizer(
#         targets,
#         truncation=True,
#         padding="max_length",
#         max_length=128,
#         return_tensors="pt",
#     )
    
#     return {
#         "input_ids": tokenized_inputs["input_ids"],
#         "attention_mask": tokenized_inputs["attention_mask"],
#         "labels": tokenized_targets["input_ids"],
#     }