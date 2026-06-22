import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import psycopg2
from sklearn.preprocessing import MinMaxScaler

# CONFIG PAGE
st.set_page_config(
    page_title="VM Stress Prediction",
    layout="wide"
)


# MODELE LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.dense = nn.Linear(64, 48)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1, :])
        return out.view(-1, 24, 2)

@st.cache_resource
def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# SIDEBAR
st.sidebar.markdown("<h1 style='color:#fff; font-size:1.6rem; margin-bottom:0;'>VM Servers</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color:#333;'>", unsafe_allow_html=True)

st.sidebar.markdown("<p style='color:#888; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Serveur</p>", unsafe_allow_html=True)
server_name = st.sidebar.selectbox("", ["VM-Inwi-01", "VM-Inwi-02", "VM-Inwi-03"], label_visibility="collapsed")

st.sidebar.markdown(f"""
    <div style='background:#1A1A1A; border:1px solid #333; border-radius:8px; padding:12px; margin-top:16px;'>
        <p style='color:#888; margin:0; font-size:0.75rem;'>ACTIF</p>
        <p style='color:white; margin:4px 0 0 0; font-weight:700; font-size:1rem;'>{server_name}</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border-color:#333; margin-top:24px;'>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    conn = psycopg2.connect(
        host="192.168.123.129",
        database="vm_monitoring",
        user="Admin",
        password="ayay2844",
        port="5432"
    )
    query = """
        SELECT timestamp, cpu_percent, ram_percent,
               heure, jour_semaine, est_weekend, vitesse_cpu
        FROM vm_metrics 
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()
df["vitesse_cpu"] = df["vitesse_cpu"].fillna(0)
scaler_cpu = MinMaxScaler()
scaler_ram = MinMaxScaler()
cpu_norm = scaler_cpu.fit_transform(df[["cpu_percent"]].values)
ram_norm = scaler_ram.fit_transform(df[["ram_percent"]].values)
data_normalise = np.hstack([cpu_norm, ram_norm])
derniere_fenetre = data_normalise[-24:]

timestamps_hist = df["timestamp"].iloc[-24:].values
cpu_hist        = df["cpu_percent"].iloc[-24:].values
ram_hist        = df["ram_percent"].iloc[-24:].values
scaler_heure   = MinMaxScaler()
scaler_jour    = MinMaxScaler()
scaler_vitesse = MinMaxScaler()
heure_norm   = scaler_heure.fit_transform(df[["heure"]].values)
jour_norm    = scaler_jour.fit_transform(df[["jour_semaine"]].values)
weekend_norm = df[["est_weekend"]].values
vitesse_norm = scaler_vitesse.fit_transform(df[["vitesse_cpu"]].values)

data_normalise = np.hstack([cpu_norm, ram_norm, heure_norm, jour_norm, weekend_norm, vitesse_norm])
# HEADER
st.markdown("<h1>Prediction des 24 prochaines heures</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# GRAPHE HISTORIQUE
st.markdown("#### Historique des 24 dernieres heures")

fig_hist = go.Figure()

fig_hist.add_trace(go.Scatter(
    x=timestamps_hist, y=cpu_hist,
    name="CPU (%)", line=dict(color="#E2001A", width=2),
    fill="tozeroy", fillcolor="rgba(226,0,26,0.08)"
))
fig_hist.add_trace(go.Scatter(
    x=timestamps_hist, y=ram_hist,
    name="RAM (%)", line=dict(color="#4A90D9", width=2),
    fill="tozeroy", fillcolor="rgba(74,144,217,0.06)"
))
fig_hist.add_hline(
    y=75, line_dash="dash", line_color="#FFA500", line_width=1,
    annotation_text="Seuil 75%", annotation_font_color="#FFA500",
    annotation_position="bottom right"
)

fig_hist.update_layout(
    plot_bgcolor="#111111",
    paper_bgcolor="#0D0D0D",
    font=dict(family="Arial", size=12, color="#CCCCCC"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", font=dict(color="#CCCCCC")),
    xaxis=dict(showgrid=True, gridcolor="#222222", color="#888888"),
    yaxis=dict(showgrid=True, gridcolor="#222222", range=[0, 100],
               title="% Utilisation", color="#888888"),
    height=300,
    margin=dict(l=40, r=40, t=20, b=40),
    dragmode=False
)

st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

# PREDICTION
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Prediction de la charge des VM sur les prochaines 24 heures")
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Lancer la prediction"):

    input_tensor = torch.tensor(derniere_fenetre, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_norm = model(input_tensor).numpy()[0]

    predictions_cpu = scaler_cpu.inverse_transform(pred_norm[:, 0].reshape(-1,1)).flatten()
    predictions_ram = scaler_ram.inverse_transform(pred_norm[:, 1].reshape(-1,1)).flatten()

    max_cpu = np.max(predictions_cpu)
    max_ram = np.max(predictions_ram)
    moy_cpu = np.mean(predictions_cpu)
    moy_ram = np.mean(predictions_ram)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Pic CPU predit",      f"{max_cpu:.1f}%")
    with col2: st.metric("Moyenne CPU predit",  f"{moy_cpu:.1f}%")
    with col3: st.metric("Pic RAM predit",      f"{max_ram:.1f}%")
    with col4: st.metric("Moyenne RAM predit",  f"{moy_ram:.1f}%")

    # Alertes
    st.markdown("<br>", unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if max_cpu > 75:
            st.error(f"Alerte CPU : Pic de {max_cpu:.1f}% predit - Risque de saturation")
        else:
            st.success(f"CPU stable - Pic predit a {max_cpu:.1f}%")
    with col_a2:
        if max_ram > 75:
            st.error(f"Alerte RAM : Pic de {max_ram:.1f}% predit - Risque de saturation")
        else:
            st.success(f"RAM stable - Pic predit a {max_ram:.1f}%")

    # Graphe combine
    st.markdown("<br>", unsafe_allow_html=True)

    cpu_futur = np.concatenate([[cpu_hist[-1]], predictions_cpu])
    ram_futur = np.concatenate([[ram_hist[-1]], predictions_ram])

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=list(range(0, 24)), y=cpu_hist,
        name="CPU Reel", line=dict(color="#E2001A", width=2)
    ))
    fig_pred.add_trace(go.Scatter(
        x=list(range(0, 24)), y=ram_hist,
        name="RAM Reel", line=dict(color="#4A90D9", width=2)
    ))
    fig_pred.add_trace(go.Scatter(
        x=list(range(23, 48)), y=cpu_futur,
        name="CPU Predit", line=dict(color="#E2001A", width=2, dash="dash")
    ))
    fig_pred.add_trace(go.Scatter(
        x=list(range(23, 48)), y=ram_futur,
        name="RAM Predit", line=dict(color="#4A90D9", width=2, dash="dash")
    ))

    fig_pred.add_vline(
        x=23, line_dash="dot", line_color="#555555",
        annotation_text="Maintenant", annotation_font_color="#888888",
        annotation_position="top"
    )
    fig_pred.add_hline(
        y=75, line_dash="dash", line_color="#FFA500", line_width=1,
        annotation_text="Seuil 75%", annotation_font_color="#FFA500",
        annotation_position="bottom right"
    )

    fig_pred.update_layout(
        plot_bgcolor="#111111",
        paper_bgcolor="#0D0D0D",
        font=dict(family="Arial", size=12, color="#CCCCCC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#CCCCCC")),
        xaxis=dict(showgrid=True, gridcolor="#222222", title="Heures", color="#888888"),
        yaxis=dict(showgrid=True, gridcolor="#222222", range=[0, 100],
                   title="% Utilisation", color="#888888"),
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        dragmode=False
    )

    st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)
    
