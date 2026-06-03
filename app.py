import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="VM Stress Prediction", layout="wide")
st.title("Prédiction du stress des VM")

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.dense = nn.Linear(64, 48) # 24h * 2 variables = 48

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1, :])      
        return out.view(-1, 24, 2)          #

# 3. Chargement du modele avec mise en cache
@st.cache_resource
def load_model():
    model = LSTMModel()
    # map_location force le chargement sur CPU pour garantir la fluidite sur le web
    model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 4. Barre laterale pour la selection de la machine
st.sidebar.header("Configuration")
server_name = st.sidebar.selectbox("Choisir un serveur", ["VM-Inwi-01", "VM-Inwi-02", "VM-Inwi-03"])

st.write(f"### Analyse en temps reel pour : {server_name}")

# 5. Chargement des donnees historiques
@st.cache_data
def load_data():
    df = pd.read_csv("Book1.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df

df = load_data()

# Preparation des scalers (Identiques a ton script d'entrainement)
scaler_cpu = MinMaxScaler()
scaler_ram = MinMaxScaler()
cpu_norm = scaler_cpu.fit_transform(df[["cpu_percent"]].values)
ram_norm = scaler_ram.fit_transform(df[["ram_percent"]].values)
data_normalise = np.hstack([cpu_norm, ram_norm])

# Extraction des 24 dernieres heures pour simuler l'etat actuel (Maintenant)
derniere_fenetre = data_normalise[-24:] 

# 6. Affichage de l'historique recent (2 graphiques superposes tres nets)
st.subheader("Historique des dernieres 24 heures")

fig_hist, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
ax1.plot(df["timestamp"].iloc[-24:], df["cpu_percent"].iloc[-24:], label="CPU (%)", color="blue")
ax1.set_ylabel("CPU %")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(df["timestamp"].iloc[-24:], df["ram_percent"].iloc[-24:], label="RAM (%)", color="green")
ax2.set_ylabel("RAM %")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig_hist)

# 7. Section de prediction via l'execution du modele
st.subheader("Previsions pour les prochaines 24 heures (CPU et RAM)")

if st.button("Lancer la prediction du futur"):
    # Preparation du tenseur d'entree pour PyTorch (batch_size = 1)
    input_tensor = torch.tensor(derniere_fenetre, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # L'inférence génère une matrice (1, 24, 2) -> on extrait le premier index [0]
        pred_norm = model(input_tensor).numpy()[0] 
        
    # Isolation et denormalisation separee pour retrouver l'echelle reelle (0-100%)
    pred_cpu_norm = pred_norm[:, 0].reshape(-1, 1)
    pred_ram_norm = pred_norm[:, 1].reshape(-1, 1)
    
    predictions_cpu = scaler_cpu.inverse_transform(pred_cpu_norm).flatten()
    predictions_ram = scaler_ram.inverse_transform(pred_ram_norm).flatten()
    
    max_cpu = np.max(predictions_cpu)
    max_ram = np.max(predictions_ram)
    
    # Affichage des alertes et indicateurs sous forme de blocs metriques
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Pic CPU predit", value=f"{max_cpu:.1f}%")
        if max_cpu > 75:
            st.error("Alerte : Risque de saturation CPU detecte")
    with col2:
        st.metric(label="Pic RAM predit", value=f"{max_ram:.1f}%")
        if max_ram > 75:
            st.error("Alerte : Risque de saturation RAM detecte")
            
    # Graphique UNIQUE regroupant les deux courbes de prediction conjointes
    fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
    ax_pred.plot(range(1, 25), predictions_cpu, label="CPU Predit (%)", color="orange", linestyle="--", marker="o")
    ax_pred.plot(range(1, 25), predictions_ram, label="RAM Predit (%)", color="red", linestyle="-.", marker="x")
    
    ax_pred.set_xlabel("Heures dans le futur (t + x)")
    ax_pred.set_ylabel("% Utilisation")
    ax_pred.set_ylim(0, 100)
    ax_pred.legend(loc="upper left")
    ax_pred.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig_pred)