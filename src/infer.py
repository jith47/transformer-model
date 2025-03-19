from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("models/trained/t5_chatbot")
tokenizer = AutoTokenizer.from_pretrained("models/trained/t5_chatbot")

def respond(user_input):
    prompt = f"respond: {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Chatbot: Hi! How can I help you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = respond(user_input)
        print(f"Bot: {response}")