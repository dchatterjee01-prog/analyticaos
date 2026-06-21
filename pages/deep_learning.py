import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR,
    TEXT_COLOR, ACCENT_COLOR
)

def _inject_css():
    st.markdown(f"""
    <style>
      .step-header {{ color: {PRIMARY_COLOR}; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid {PRIMARY_COLOR}33; padding-bottom: 0.3rem; }}
      .arch-box {{ background: {SURFACE_COLOR}; border: 1px dashed {ACCENT_COLOR}; padding: 1rem; border-radius: 8px; text-align: center; font-family: monospace; color: {TEXT_COLOR}; margin-bottom: 1rem; }}
      .metric-card {{ background: {SURFACE_COLOR}; border: 1px solid {PRIMARY_COLOR}33; border-radius: 10px; padding: 1rem; text-align: center; }}
      .metric-val {{ font-size: 1.6rem; font-weight: 800; color: {ACCENT_COLOR}; }}
      .metric-lbl {{ font-size: 0.72rem; color: {TEXT_COLOR}88; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }}
    </style>
    """, unsafe_allow_html=True)

def _metric_card(col, value, label):
    col.markdown(f'<div class="metric-card"><div class="metric-val">{value}</div><div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

# ── PYTORCH MODEL ARCHITECTURE ────────────────────────────────────────────────
class TabularNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, output_dim: int):
        super(TabularNN, self).__init__()
        layers = []
        curr_dim = input_dim
        
        for h in hidden_layers:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.2))
            curr_dim = h
            
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ── MAIN UI ───────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("⚡ Deep Learning Engine")
    st.caption("Phase 13 — Custom PyTorch Neural Networks for Tabular Data")

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        return

    # --- STEP 1 & 2: CONFIGURATION & PREPROCESSING ---
    st.markdown('<div class="step-header">1. Architecture & Data Setup</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    target_col = c1.selectbox("Target Column:", options=df.columns, key="dl_target")
    
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    unique_vals = df[target_col].nunique()
    default_task = "Classification" if (not is_numeric or unique_vals < 20) else "Regression"
    task_type = c2.radio("Task Type:", ["Classification", "Regression"], index=0 if default_task == "Classification" else 1, horizontal=True)
    is_class = task_type == "Classification"

    feature_cols = st.multiselect(
        "Feature Columns (Inputs):", 
        options=[c for c in df.select_dtypes(include="number").columns if c != target_col],
        default=[c for c in df.select_dtypes(include="number").columns if c != target_col][:10],
        key="dl_features"
    )

    if not feature_cols:
        st.info("Please select at least one numeric feature column.")
        return

    h1, h2, h3 = st.columns(3)
    hidden_input = h1.text_input("Hidden Layers:", value="128, 64", help="Comma separated neurons")
    epochs = h2.number_input("Epochs:", 10, 2000, 200, 50)
    lr = h3.number_input("Learning Rate:", 0.0001, 0.1, 0.005, 0.001, format="%.4f")

    try:
        hidden_layers = [int(x.strip()) for x in hidden_input.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid hidden layer format.")
        return

    input_dim = len(feature_cols)
    output_dim = df[target_col].nunique() if (is_class and df[target_col].nunique() > 2) else 1

    arch_str = f"Input({input_dim}) ➔ " + " ➔ ".join([f"Linear({h}) ➔ ReLU ➔ Dropout" for h in hidden_layers]) + f" ➔ Output({output_dim})"
    st.markdown(f'<div class="arch-box"><b>Architecture:</b><br>{arch_str}</div>', unsafe_allow_html=True)

    if st.button("⚙️ Process Data", key="dl_prep"):
        model_df = df[feature_cols + [target_col]].dropna().copy()
        X = model_df[feature_cols]
        y = model_df[target_col]

        X_scaled = StandardScaler().fit_transform(X)

        if is_class:
            y_encoded = LabelEncoder().fit_transform(y)
        else:
            y_encoded = y.values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        st.session_state["dl_tensors"] = {
            "X_train": torch.tensor(X_train, dtype=torch.float32),
            "X_test": torch.tensor(X_test, dtype=torch.float32),
            "y_train": torch.tensor(y_train, dtype=torch.long if is_class else torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.long if is_class else torch.float32),
        }
        st.session_state["dl_config"] = {"in": input_dim, "hid": hidden_layers, "out": output_dim, "is_class": is_class, "ep": epochs, "lr": lr}
        st.success("Data processed and converted to PyTorch Tensors. Ready to train.")

    # --- STEP 3: TRAINING LOOP ---
    if "dl_tensors" in st.session_state:
        st.markdown('<div class="step-header">2. PyTorch Training Loop</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Train Neural Network", type="primary"):
            t = st.session_state["dl_tensors"]
            cfg = st.session_state["dl_config"]

            model = TabularNN(cfg["in"], cfg["hid"], cfg["out"])
            criterion = nn.CrossEntropyLoss() if cfg["is_class"] and cfg["out"] > 1 else (nn.BCEWithLogitsLoss() if cfg["is_class"] else nn.MSELoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_history = []

            # Training Loop
            model.train()
            for epoch in range(cfg["ep"]):
                optimizer.zero_grad()
                outputs = model(t["X_train"])
                
                if cfg["out"] == 1:
                    outputs = outputs.squeeze()
                
                loss = criterion(outputs, t["y_train"])
                loss.backward()
                optimizer.step()
                
                loss_history.append(loss.item())
                
                if epoch % max(1, cfg["ep"] // 20) == 0 or epoch == cfg["ep"] - 1:
                    progress_bar.progress((epoch + 1) / cfg["ep"])
                    status_text.text(f"Training Epoch {epoch+1}/{cfg['ep']}... Loss: {loss.item():.4f}")

            status_text.text(f"✅ Training Complete. Final Loss: {loss_history[-1]:.4f}")

            # --- STEP 4: EVALUATION ---
            model.eval()
            with torch.no_grad():
                test_outputs = model(t["X_test"])
                if cfg["out"] == 1:
                    test_outputs = test_outputs.squeeze()

                y_true = t["y_test"].numpy()
                
                st.markdown('<div class="step-header">3. Model Evaluation (Holdout Set)</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)

                if cfg["is_class"]:
                    if cfg["out"] > 1:
                        y_pred = torch.argmax(test_outputs, dim=1).numpy()
                    else:
                        y_pred = (torch.sigmoid(test_outputs) > 0.5).long().numpy()
                    
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average="weighted")
                    _metric_card(m1, f"{acc:.2%}", "Accuracy")
                    _metric_card(m2, f"{f1:.4f}", "Weighted F1 Score")
                else:
                    y_pred = test_outputs.numpy()
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    _metric_card(m1, f"{r2:.4f}", "R² Score")
                    _metric_card(m2, f"{rmse:.4f}", "RMSE")
                    _metric_card(m3, f"{mae:.4f}", "MAE")

            # Plot Loss Curve
            fig = go.Figure(go.Scatter(y=loss_history, mode="lines", name="Training Loss", line=dict(color=PRIMARY_COLOR, width=2.5)))
            fig.update_layout(title="Convergence (Loss Curve)", xaxis_title="Epoch", yaxis_title="Loss", paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=SURFACE_COLOR, font_color=TEXT_COLOR, margin=dict(t=40, b=40), height=350)
            st.plotly_chart(fig, width="stretch")