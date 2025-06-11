# 🛡️ Text Shield: Privacy-Preserving Federated Learning for Text Classification

> **"Training smarter models, without ever touching your data."**  
> A federated learning simulation built from scratch to protect user privacy in NLP tasks.

---

## 🚀 Project Overview

**Text Shield** is a robust demonstration of **Federated Learning (FL)** applied to a multi-client text classification task. Designed with **privacy**, **scalability**, and **real-world deployment** in mind, this project simulates how edge devices collaboratively train a shared model **without centralizing their data**.

This isn't just a tech demo — it's a proof of concept for the **future of privacy-first machine learning**.

---

## 🧠 Problem Statement

Traditional machine learning requires aggregating all user data to a centralized server, introducing major **privacy risks** and **regulatory challenges** (like GDPR and HIPAA).

**Text Shield solves this problem by training on decentralized datasets.** Each client (or "user") retains local control of their data while participating in collaborative model improvement.

---

## 🏗️ Architecture Overview

```plaintext
 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │  Client 1   │    │  Client 2   │    │  Client 3   │
 │ Local Data  │    │ Local Data  │    │ Local Data  │
 └────┬────────┘    └────┬────────┘    └────┬────────┘
      │                  │                  │
      ▼                  ▼                  ▼
  [Local Training]   [Local Training]   [Local Training]
      │                  │                  │
      └────┬────────┬────┴────┬────────┬────┘
           ▼        ▼         ▼
     🔁 [Server Aggregation - FedAvg]
           │
           ▼
   📦 Updated Global Model
````

---

## 🔄 Federated Learning Workflow

1. **Initialize Global Model** on the server.
2. For each round:

   * Clients receive the current global weights.
   * Each client trains on **local text data**.
   * Clients send **updated weights** (not data!) to the server.
   * The server aggregates weights via **Federated Averaging (FedAvg)**.
3. Evaluate the final model on a **central test set**.

---

## 📊 Results & Performance

### 🧪 Final Evaluation Metrics

* **Test Accuracy:** 87.82%
* **Test Loss:** 0.3437
* **Rounds Trained:** 5
* **Clients Simulated:** 3

### 📈 Accuracy Per Round

```plaintext
Round 1 - Accuracy: ~70%
Round 2 - Accuracy: ~77%
Round 3 - Accuracy: ~82%
Round 4 - Accuracy: ~85%
Round 5 - Accuracy: ~87%
```

*(Visuals can be added with Matplotlib or TensorBoard)*

---


## 🎥 Live Demo: Text Shield in Action

![Text Shield Federated Learning Demo](./output_demo_video/fl_demo_output.gif)

> 🛡️ *Watch how local clients train independently and the server smartly aggregates the knowledge — all without touching private data.*  
> Welcome to the future of **privacy-first machine learning**.

---


## 🧾 Sample Predictions

```
Input: "This product is amazing and I love it!"
Prediction: Positive (✅)

Input: "I regret buying this. Waste of money."
Prediction: Negative (❌)
```

> ⚠️ Note: Sample predictions based on synthetic dataset simulation for testing.

---

## 💾 Model Export

The final trained model is saved as:

```python
model.save("global_model.h5")
```

✅ Also convertible to **TensorFlow Lite** for mobile deployment.

---

## 📦 Project Structure

```
FL_demo/
├── client_data/           # Simulated datasets for each client
├── fl_model.py            # Model architecture & compilation
├── server.py              # FL orchestration logic
├── utils.py               # Preprocessing & helper functions
├── requirements.txt       # Python packages
├── global_model.h5        # Saved final model
└── README.md              # This file!
```

---

## 💡 Key Learnings

* Hands-on understanding of **Federated Averaging**
* Importance of **privacy-aware training**
* Trade-offs in communication vs computation
* Simulating **non-IID client data** environments

---

## 🔧 Requirements

```bash
Python 3.12
TensorFlow >= 2.10
NumPy
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Author

**Selcii** —
AI & Data Science Enthusiast | HealthTech Builder
*“Building the future, one neural net at a time.”*

🔗 [LinkedIn](https://linkedin.com/in/maria-selciya-m) •  🧠 (https://github.com/MICHAEL-MARIA16/federated_learning_demo)

---

## ⭐ Show Some Love

If you found this project valuable:

🌟 Star it on GitHub
🍴 Fork it
📣 Share it with your ML crew

> Let's build a future where **privacy is the default**, not a privilege.

---
