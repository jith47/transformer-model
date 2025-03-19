import tensorflow as tf
import numpy as np
import json

class TFLitePredictor:
    def __init__(self, tflite_path, label_to_index):
        self.interpreter = tf.lite.Interpreter(tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.label_to_index = label_to_index
        self.index_to_label = {v: k for k, v in label_to_index.items()}
        self.max_length = 10  # Match the training sequence length
    
    def predict(self, text):
        # Preprocess input text
        text = text.lower().strip()
        tokenized = [1 if word in text else 0 for word in ["hi", "hello", "weather", "rain"]]
        
        # Pad the sequence to max_length
        padded = np.zeros(self.max_length, dtype=np.int32)  # Use INT32
        padded[:len(tokenized)] = tokenized  # Fill with tokenized values
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], [padded])
        self.interpreter.invoke()
        
        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        intent_idx = np.argmax(output)
        return self.index_to_label[intent_idx]

if __name__ == "__main__":
    # Load label-to-index mapping
    with open("data/raw/Intent.json", "r") as f:
        data = json.load(f)
    label_to_index = {intent_data["intent"]: idx for idx, intent_data in enumerate(data["intents"])}
    
    # Load the TFLite model
    predictor = TFLitePredictor("models/trained/chatbot.tflite", label_to_index)
    
    # Interactive prompt
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        intent = predictor.predict(user_input)
        print(f"Bot: Detected intent - {intent}")