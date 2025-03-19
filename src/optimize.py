import tensorflow as tf
import tensorflow_model_optimization as tfmot

def prune_model(model_path, save_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=0.5,  # Prune 50% of weights
            begin_step=0,         # Start pruning from step 0
            end_step=-1,          # Continue pruning until the end of training
            frequency=100         # Apply pruning every 100 steps
        )
    }
    
    # Apply pruning to the model
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruned_model = prune_low_magnitude(model, **pruning_params)
    
    # Re-compile the pruned model
    pruned_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the pruned model
    pruned_model.save(save_path)
    print(f"Pruned model saved to {save_path}")

if __name__ == "__main__":
    prune_model("models/trained/chatbot", "models/trained/chatbot_pruned")