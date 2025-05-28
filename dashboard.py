import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

st.set_page_config(page_title="🛡️ Text Shield Dashboard", layout="wide")

st.title("🛡️ Text Shield - Federated Learning Dashboard")
st.markdown("Visualizing decentralized training. One round at a time.")

# ========== Load training log ==========
LOG_PATH = "training_logs.json"

if not os.path.exists(LOG_PATH):
    st.warning("⚠️ No training log found. Please run the training and save logs to `training_logs.json`.")
    st.stop()

with open(LOG_PATH, "r") as f:
    log_data = json.load(f)

rounds = log_data["rounds"]
accuracy = log_data["accuracy"]
loss = log_data["loss"]
client_stats = log_data["client_stats"]
sample_predictions = log_data.get("sample_predictions", [])

# ========== Global Accuracy Graph ==========
st.subheader("📈 Global Model Accuracy Over Rounds")
fig_acc, ax1 = plt.subplots()
sns.lineplot(x=rounds, y=accuracy, marker="o", ax=ax1)
ax1.set_xlabel("Round")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1)
st.pyplot(fig_acc)

# ========== Global Loss Graph ==========
st.subheader("📉 Global Model Loss Over Rounds")
fig_loss, ax2 = plt.subplots()
sns.lineplot(x=rounds, y=loss, marker="o", color="red", ax=ax2)
ax2.set_xlabel("Round")
ax2.set_ylabel("Loss")
st.pyplot(fig_loss)

# ========== Client Training Stats ==========
st.subheader("👥 Client-wise Training Stats")

df_clients = pd.DataFrame(client_stats)
st.dataframe(df_clients.style.format({"accuracy": "{:.2%}", "loss": "{:.4f}"}))

# ========== Sample Predictions ==========
st.subheader("🧾 Sample Predictions")

if sample_predictions:
    for pred in sample_predictions:
        st.markdown(f"- **Input:** {pred['text']}")
        st.markdown(f"  - 🔍 **Predicted:** `{pred['prediction']}`")
        st.markdown("---")
else:
    st.info("No sample predictions logged yet. You can modify server.py to add them.")

# ========== Footer ==========
st.markdown("---")
st.markdown("🚀 Built by **Selcii** | Federated, Secure, Future-Proof AI 🌍")

