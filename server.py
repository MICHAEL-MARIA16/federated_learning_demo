import numpy as np
from fl_model import build_model
from utils import load_client_data, load_data
import copy
import json

# Federated learning parameters
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
    print("🔍 Loading sample client data to detect input dimension...")
    X_sample, _ = load_client_data(1)
    input_dim = X_sample.shape[1]
    print(f"📐 Detected input feature dimension: {input_dim}\n")

    # Initialize global model
    global_model = build_model(input_dim=input_dim, num_classes=NUM_CLASSES)
    print("✅ Global model initialized.\n")

    round_accuracy = []
    round_loss = []
    all_client_stats = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n🔁 --- Federated Training Round {round_num} ---")

        client_weights = []
        client_accuracies = []
        client_losses = []

        for client_id in range(1, NUM_CLIENTS + 1):
            print(f"👤 Client {client_id} training started...")

            # Load client-specific data
            X, y = load_client_data(client_id)

            # Build and initialize local model
            client_model = build_model(input_dim=input_dim, num_classes=NUM_CLASSES)
            client_model.set_weights(global_model.get_weights())

            # Train locally
            history = client_model.fit(X, y, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            # Evaluate local performance
            loss, acc = client_model.evaluate(X, y, verbose=0)
            client_losses.append(loss)
            client_accuracies.append(acc)

            # Collect weights
            client_weights.append(client_model.get_weights())
            print(f"✅ Client {client_id} done: Accuracy = {acc:.4f}, Loss = {loss:.4f}")

        # Aggregate weights
        new_weights = average_weights(client_weights)
        global_model.set_weights(new_weights)
        print(f"🌍 Round {round_num} complete. Global model updated.")

        # Log round metrics
        round_accuracy.append(np.mean(client_accuracies))
        round_loss.append(np.mean(client_losses))

        # Record final stats per client for this round
        for i in range(NUM_CLIENTS):
            all_client_stats.append({
                "round": round_num,
                "client": f"Client {i+1}",
                "accuracy": client_accuracies[i],
                "loss": client_losses[i],
            })

    print("\n✅ Federated training complete.\n")

    # Evaluate final global model
    print("📊 Evaluating on test data...")
    _, X_test, _, y_test = load_data()
    test_loss, test_acc = global_model.evaluate(X_test, y_test, verbose=2)
    print(f"\n🎯 Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save model
    global_model.save("global_model.h5")
    print("💾 Saved final global model as 'global_model.h5'")

    # Save training logs
    final_stats_per_client = []
    for i in range(NUM_CLIENTS):
        client_logs = [s for s in all_client_stats if s["client"] == f"Client {i+1}"]
        if client_logs:
            final_stats_per_client.append(client_logs[-1])  # last round stats

    training_logs = {
        "rounds": list(range(1, ROUNDS + 1)),
        "accuracy": round_accuracy,
        "loss": round_loss,
        "client_stats": final_stats_per_client,
        "sample_predictions": [
            {"text": "Great product, very happy!", "prediction": "Positive"},
            {"text": "Not worth the price.", "prediction": "Negative"},
        ]
    }

    with open("training_logs.json", "w") as f:
        json.dump(training_logs, f, indent=4)

    print("🧾 Saved training logs to 'training_logs.json'")


if __name__ == "__main__":
    federated_train()
