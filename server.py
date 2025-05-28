import numpy as np
from fl_model import build_model
from utils import load_client_data
import copy

NUM_CLIENTS = 3
NUM_CLASSES = 3
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
ROUNDS = 5


def average_weights(weight_list):
    """Averages a list of model weights."""
    avg_weights = []
    for weights in zip(*weight_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights


def federated_train():
    print("Loading sample client data to detect input dimension...")
    X_sample, _ = load_client_data(1)
    input_dim = X_sample.shape[1]
    print(f"Detected input feature dimension: {input_dim}")

    # Initialize global model
    global_model = build_model(input_dim=input_dim, num_classes=NUM_CLASSES)
    print("✅ Global model initialized.\n")

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Federated Training Round {round_num} ---")

        client_weights = []

        for client_id in range(1, NUM_CLIENTS + 1):
            print(f"Client {client_id} training started...")

            # Load client data
            X, y = load_client_data(client_id)

            # Create local model as a copy of the global model
            client_model = build_model(input_dim=input_dim, num_classes=NUM_CLASSES)
            client_model.set_weights(global_model.get_weights())

            # Train local model on client data
            client_model.fit(X, y, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            # Collect client weights
            client_weights.append(client_model.get_weights())

            print(f"Client {client_id} training completed.")

        # Aggregate weights (simple average)
        new_weights = average_weights(client_weights)

        # Update global model weights
        global_model.set_weights(new_weights)

        print(f"Round {round_num} complete. Global model updated.\n")

    print("Federated training finished.")
    print("Evaluating global model on test data...")

    # Optionally, load test data and evaluate global model
    from utils import load_data
    _, X_test, _, y_test = load_data()
    results = global_model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")


if __name__ == "__main__":
    federated_train()
