import tensorflow as tf

def convert(keras_model_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(keras_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    convert("models/trained/chatbot_pruned", "models/trained/chatbot.tflite")