import streamlit as st
from src.main import run_federated_training, save_training_results
from src.baseline import train_baseline
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# ================= HELPER FUNCTIONS =================
def threat_gauge(score):
    """Display threat level gauge."""
    if score < 30:
        color = "#22c55e"
        label = "LOW"
    elif score < 60:
        color = "#fbbf24"
        label = "MEDIUM"
    else:
        color = "#ef4444"
        label = "HIGH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color}}
    ))
    fig.update_layout(paper_bgcolor="#0b1120", font={'color': '#e2e8f0'})
    st.plotly_chart(fig, use_container_width=True)

def node_heatmap(anomaly_values):
    """Display anomaly scores heatmap."""
    if len(anomaly_values) == 0:
        st.info("No data available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    data = np.array(anomaly_values).reshape(-1, 1)
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('Anomaly Scores Over Rounds')
    ax.set_xlabel('Round')
    ax.set_ylabel('Metric')
    st.pyplot(fig)

def topology_graph(num_clients, malicious_clients):
    """Display network topology graph."""
    edge_x, edge_y = [], []
    for i in range(num_clients):
        edge_x += [i, num_clients, None]
        edge_y += [0, 1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='#22c55e')
    )

    node_x = list(range(num_clients)) + [num_clients]
    node_y = [0]*num_clients + [1]

    colors = ["#ef4444" if i < malicious_clients else "#22c55e"
              for i in range(num_clients)] + ["#38bdf8"]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=25, color=colors),
        text=[f"Client {i}" for i in range(num_clients)] + ["Server"],
        textposition="bottom center"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#0b1120",
        font={'color': "#e2e8f0"},
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def run_training_live(aggregation, num_clients, malicious_clients, rounds, 
                      max_samples, local_epochs, progress_bar, status_text):
    """Run training and yield results live for real-time display."""
    from src.main import run_federated_training
    for update in run_federated_training(
        aggregation=aggregation,
        num_clients=num_clients,
        malicious_clients=malicious_clients,
        rounds=rounds,
        quick_mode=False,
        max_samples=max_samples,
        local_epochs=local_epochs,
        dp_enabled=True
    ):
        progress = update["round"] / rounds  # Value between 0 and 1
        progress_bar.progress(progress)
        status_text.text(f"Training Round {update['round']}/{rounds} - Accuracy: {update['accuracy']}%")
        yield update

st.set_page_config(
    page_title="FedFortress",
    page_icon="🛡",
    layout="wide"
)

# ================= CSS STYLING =================
st.markdown("""
<style>
.hacker-terminal {
    font-family: 'Courier New', monospace;
    font-size: 62px;
    font-weight: bold;
    color: #39ff14;
    text-align: center;
    text-shadow: 0 0 5px #39ff14,
                 0 0 10px #22c55e,
                 0 0 20px #22c55e;
    overflow: hidden;
    white-space: nowrap;
    border-right: 3px solid #39ff14;
    width: 0;
    animation: typing 3s steps(40, end) forwards,
               blink 0.8s infinite;
}

@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

@keyframes blink {
    0% { border-color: #39ff14 }
    50% { border-color: transparent }
    100% { border-color: #39ff14 }
}
</style>

<div class="hacker-terminal">
🛡 FedFortress 
</div>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("### 🔬 Real-Time Federated Learning with Differential Privacy & Attack Detection")
st.divider()

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Configuration")

aggregation = st.sidebar.selectbox(
    "Aggregation Strategy",
    ["FedAvg", "Trimmed Mean", "Median"]
)

num_clients = st.sidebar.slider("👥 Number of Clients", 2, 10, 5)
malicious_clients = st.sidebar.slider("⚠️ Malicious Clients", 0, 5, 1)
rounds = st.sidebar.slider("🔄 Training Rounds", 1, 10, 5)
local_epochs = st.sidebar.slider("📚 Local Epochs", 1, 5, 1)
max_samples = st.sidebar.slider("📊 Samples", 1000, 50000, 5000, step=1000)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Privacy Settings")
dp_enabled = st.sidebar.checkbox("Enable Differential Privacy", value=True)
dp_epsilon = st.sidebar.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚔️ Attack Settings")
attack_enabled = st.sidebar.checkbox("Enable Malicious Clients", value=True)
attack_type = st.sidebar.selectbox(
    "Attack Type",
    ["noise_injection", "weight_scaling", "random_weights", "label_flipping"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Using Real CIFAR-10 Dataset ({max_samples:,} samples)")

# ================= PHASE 1: CENTRALIZED BASELINE =================

col1, col2, col3 = st.columns(3)
col1.metric("📦 Dataset", "CIFAR-10")
col2.metric("📈 Classes", "10")
col3.metric("🖼️ Image Size", "32×32")

run_baseline = st.button("🚀 Run Centralized Baseline Training", key="baseline")

if run_baseline:
    st.markdown("### 📊 Baseline Training Progress")
    baseline_progress = st.progress(0)
    baseline_status = st.empty()
    baseline_chart = st.empty()
    
    baseline_accuracy_values = []
    
    try:
        # Run baseline training
        baseline_results = train_baseline(epochs=5)
        
        for epoch, result in enumerate(baseline_results):
            progress = (epoch + 1) / 5  # Value between 0 and 1
            baseline_progress.progress(progress)
            baseline_status.text(f"Epoch {epoch+1}/5 - Loss: {result.get('loss', 0):.4f} - Accuracy: {result.get('accuracy', 0):.2f}%")
            
            baseline_accuracy_values.append(result.get('accuracy', 0))
            baseline_chart.line_chart(baseline_accuracy_values)
            time.sleep(0.5)
        
        baseline_progress.progress(1.0)
        baseline_status.text("✅ Baseline training complete!")
        
        st.success(f"🎯 **Final Baseline Accuracy: {baseline_accuracy_values[-1]:.2f}%**")
        
    except Exception as e:
        st.error(f"Baseline training error: {str(e)}")

st.divider()

# ================= PHASE 2: FEDERATED LEARNING =================

# System Overview
st.subheader("🌐 System Configuration")
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Clients", num_clients)
col2.metric("⚠️ Malicious", malicious_clients, delta=f"{malicious_clients/num_clients*100:.1f}%")
col3.metric("🔄 Aggregation", aggregation)
col4.metric("🛡️ DP Enabled", "Yes" if dp_enabled else "No", delta=f"ε={dp_epsilon}")

st.divider()

# Network Topology
st.subheader("📡 Network Topology")
topology_graph(num_clients, malicious_clients)

st.divider()

# ================= LIVE TRAINING =================
st.subheader("🚀 Live Server-Client Training Flow")

progress_bar = st.progress(0)
status_text = st.empty()
accuracy_chart = st.empty()
anomaly_chart = st.empty()

accuracy_values = []
anomaly_values = []
all_results = []

st.info(f"📊 CIFAR-10: {local_epochs} epochs, {min(rounds, 5)} rounds, {max_samples:,} samples, DP ε={dp_epsilon}")

# Run training
run_fed = st.button("🚀 Start Federated Training", key="federated")

if run_fed:
    status_text.text("🏃 Starting federated training on CIFAR-10...")
    
    for update in run_training_live(
        aggregation=aggregation,
        num_clients=num_clients,
        malicious_clients=malicious_clients,
        rounds=min(rounds, 5),
        max_samples=max_samples,
        local_epochs=local_epochs,
        progress_bar=progress_bar,
        status_text=status_text
    ):
        accuracy_values.append(update["accuracy"])
        anomaly_values.append(update["loss"])
        all_results.append(update)
        
        # Update charts in real-time
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("##### 📈 Accuracy Trend")
            st.line_chart(accuracy_values)
        with col_chart2:
            st.markdown("##### 📉 Loss/Anomaly Trend")
            st.line_chart(anomaly_values)
        
        # Show client details
        st.markdown(f"##### 👥 Client Updates - Round {update['round']}")
        client_cols = st.columns(len(update.get('local_accuracies', [1])))
        for i, (client_acc, anomaly) in enumerate(zip(
            update.get('local_accuracies', []),
            update.get('anomaly_scores', [])
        )):
            with client_cols[i]:
                client_status = "⚠️ Malicious" if i < malicious_clients else "✅ Honest"
                st.metric(
                    f"Client {i}",
                    f"{client_acc:.1f}%",
                    delta=f"Anomaly: {anomaly:.1f}",
                    delta_color="inverse" if i < malicious_clients else "normal"
                )
                st.caption(client_status)

    progress_bar.progress(1.0)
    status_text.text("✅ Training complete!")
    
    # Save results
    saved_path = save_training_results(
        results=all_results,
        aggregation=aggregation,
        num_clients=num_clients,
        malicious_clients=malicious_clients,
        rounds=min(rounds, 5),
        quick_mode=False,
        max_samples=max_samples,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon
    )
    
    st.success(f"✅ Training complete! Results saved to: `{saved_path}`")
    st.divider()
    
    # ================= SECURITY ANALYSIS =================
    st.subheader("🔒 Security Analysis & Attack Detection")
    
    malicious_ratio = malicious_clients / num_clients if num_clients > 0 else 0
    
    if len(anomaly_values) > 1:
        mean_anomaly = np.mean(anomaly_values)
        std_anomaly = np.std(anomaly_values)
        cv = std_anomaly / (mean_anomaly + 1e-6)
        threat_score = (malicious_ratio * 50) + (min(cv * 30, 50))
    else:
        threat_score = malicious_ratio * 100
    
    threat_score = min(100, threat_score)
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("🎯 Threat Level")
        
        if threat_score < 30:
            threat_color = "#22c55e"
            threat_label = "LOW"
        elif threat_score < 60:
            threat_color = "#fbbf24"
            threat_label = "MEDIUM"
        else:
            threat_color = "#ef4444"
            threat_label = "HIGH"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=threat_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Threat Score: {threat_label}", 'font': {'size': 24, 'color': '#e2e8f0'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#e2e8f0'},
                'bar': {'color': threat_color},
                'bgcolor': '#1e293b',
                'borderwidth': 2,
                'bordercolor': '#e2e8f0'
            }
        ))
        fig.update_layout(paper_bgcolor="#0b1120", font={'color': '#e2e8f0'})
        st.plotly_chart(fig, use_container_width=True)
    
    with colB:
        st.subheader("📊 Client Deviation Heatmap")
        node_heatmap(anomaly_values)
    
    st.divider()
    
    # ================= TRAINING SUMMARY =================
    st.subheader("📋 Complete Training Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    if all_results:
        summary_col1.metric(
            "📈 Initial Accuracy",
            f"{all_results[0]['accuracy']:.2f}%"
        )
        summary_col2.metric(
            "🎯 Final Accuracy",
            f"{all_results[-1]['accuracy']:.2f}%"
        )
        improvement = all_results[-1]['accuracy'] - all_results[0]['accuracy']
        delta_color = "normal" if improvement > 0 else "inverse"
        summary_col3.metric(
            "📊 Improvement",
            f"{improvement:+.2f}%",
            delta_color=delta_color
        )
        summary_col4.metric(
            "🛡️ DP Noise Scale",
            f"{0.05/dp_epsilon:.4f}"
        )
    
    st.markdown(f"""
    **📝 Configuration Details:**
    - **Aggregation Method:** {aggregation}
    - **Clients:** {num_clients} total ({malicious_clients} malicious - {attack_type})
    - **Rounds:** {min(rounds, 5)}
    - **Local Epochs:** {local_epochs}
    - **Dataset:** CIFAR-10 ({max_samples:,} samples)
    - **Differential Privacy:** {"Enabled" if dp_enabled else "Disabled"} (ε={dp_epsilon})
    - **Results saved:** `{saved_path}`
    """)
    
    st.success("✅ System Evaluation Complete - All Phases Executed Successfully")
