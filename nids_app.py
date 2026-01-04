# ============================================
# AI-Based Network Intrusion Detection System
# Final Custom Project Version
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI NIDS | Cybersecurity Project",
    layout="wide"
)

st.title("🔐 AI-Based Network Intrusion Detection System")
st.markdown("""
This project demonstrates how **Machine Learning** can be used to  
detect **malicious network activity** using traffic behavior patterns.
""")

# -----------------------------
# Data Generation
# -----------------------------
@st.cache_data
def generate_network_data():
    np.random.seed(1)
    samples = 4000

    data = {
        "port_number": np.random.randint(1, 65535, samples),
        "connection_time": np.random.randint(1, 100000, samples),
        "packet_count": np.random.randint(1, 300, samples),
        "avg_packet_size": np.random.uniform(40, 1500, samples),
        "idle_time": np.random.uniform(0, 1000, samples),
        "attack": np.random.choice([0, 1], samples, p=[0.75, 0.25])
    }

    df = pd.DataFrame(data)

    # Introduce attack behavior
    df.loc[df["attack"] == 1, "packet_count"] += np.random.randint(100, 300)
    df.loc[df["attack"] == 1, "connection_time"] = np.random.randint(1, 500)

    return df

df = generate_network_data()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Model Controls")

train_ratio = st.sidebar.slider("Training Data (%)", 60, 90, 80)
trees = st.sidebar.slider("Random Forest Trees", 50, 200, 100)

# -----------------------------
# Data Preparation
# -----------------------------
X = df.drop("attack", axis=1)
y = df["attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - train_ratio) / 100, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
st.subheader("1️⃣ Train Intrusion Detection Model")

if st.button("🚀 Train Model"):
    with st.spinner("Training the AI model..."):
        model = RandomForestClassifier(n_estimators=trees)
        model.fit(X_train, y_train)
        st.session_state["model"] = model

    st.success("Model trained successfully!")

# -----------------------------
# Evaluation Section
# -----------------------------
st.subheader("2️⃣ Model Performance")

if "model" in st.session_state:
    model = st.session_state["model"]
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy * 100:.2f} %")
    c2.metric("Total Records", len(df))
    c3.metric("Detected Attacks", predictions.sum())

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=False,
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=False)

else:
    st.info("Please train the model to view performance metrics.")

# -----------------------------
# Live Traffic Testing
# -----------------------------
st.divider()
st.subheader("3️⃣ Live Network Traffic Analyzer")

col1, col2, col3, col4, col5 = st.columns(5)

port = col1.number_input("Port Number", 1, 65535, 80)
duration = col2.number_input("Connection Time", 1, 100000, 400)
packets = col3.number_input("Packet Count", 1, 600, 120)
packet_size = col4.number_input("Avg Packet Size", 40, 1500, 600)
idle = col5.number_input("Idle Time", 0, 1000, 50)

if st.button("🔍 Analyze Traffic"):
    if "model" in st.session_state:
        input_data = np.array([[port, duration, packets, packet_size, idle]])
        result = st.session_state["model"].predict(input_data)

        if result[0] == 1:
            st.error("🚨 ALERT: Malicious Network Activity Detected!")
            st.write("Possible causes: High packet rate or abnormal connection duration.")
        else:
            st.success("✅ Traffic is Normal and Safe.")
    else:
        st.warning("Train the model before analyzing traffic.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(" AI in Cybersecurity | Network Intrusion Detection System")
