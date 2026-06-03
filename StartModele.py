import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\monitoring_vm_6mois.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
cpu = df["cpu_percent"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(cpu)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.dense = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1, :])
        return out

# 4. Charger les poids sauvegardes
model = LSTMModel().to(device)
model.load_state_dict(torch.load(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\lstm_model.pth"))
model.eval()
print("Modele charge !")

# 5. Preparer les donnees test (20% de la fin)
FENETRE = 24
cpu_norm = scaler.transform(cpu)

X, y = [], []
for i in range(len(cpu_norm) - FENETRE):
    X.append(cpu_norm[i:i+FENETRE])
    y.append(cpu_norm[i+FENETRE])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

split = int(len(X) * 0.8)
X_test = torch.tensor(X[split:]).to(device)
y_test = torch.tensor(y[split:]).to(device)

# 6. Predire sur le test
with torch.no_grad():
    predictions_norm = model(X_test).cpu().numpy()

predictions = scaler.inverse_transform(predictions_norm)
y_test_reel = scaler.inverse_transform(y_test.cpu().numpy())

# 7. Graphe Reel vs Predit
plt.figure(figsize=(14, 5))
plt.plot(y_test_reel[:200], label="CPU Reel",   color="blue")
plt.plot(predictions[:200],  label="CPU Predit", color="orange", linestyle="--")
plt.title("Prediction CPU - LSTM (200 premieres heures du test)")
plt.xlabel("Heures")
plt.ylabel("CPU %")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\prediction_cpu.png")
plt.show()

# 8. Predire la prochaine heure
dernieres_24h = cpu[-24:]
dernieres_24h_norm = scaler.transform(dernieres_24h)
X_prochain = torch.tensor(dernieres_24h_norm, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred_norm = model(X_prochain).cpu().numpy()

pred = scaler.inverse_transform(pred_norm)
print(f"CPU predit prochaine heure : {pred[0][0]:.2f}%")

# 9. Pics et creux
SEUIL_PIC   = 75
SEUIL_CREUX = 35
mae = np.mean(np.abs(predictions - y_test_reel))
pics  = np.where(predictions > SEUIL_PIC)[0]
creux = np.where(predictions < SEUIL_CREUX)[0]

print(f"Pics detectes   (CPU > {SEUIL_PIC}%) : {len(pics)} heures")
print(f"Creux detectes  (CPU < {SEUIL_CREUX}%) : {len(creux)} heures")
print(f"Erreur moyenne (MAE) : {mae:.2f}%")