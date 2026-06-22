import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import psycopg2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilise : {device}")

conn = psycopg2.connect(
    host="192.168.123.129",
    database="vm_monitoring",
    user="Admin",
    password="ayay2844",
    port="5432"
)

df = pd.read_sql("""
    SELECT timestamp, cpu_percent, ram_percent, 
           heure, jour_semaine, est_weekend, vitesse_cpu
    FROM vm_metrics 
    ORDER BY timestamp
""", conn)
conn.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["vitesse_cpu"] = df["vitesse_cpu"].fillna(0)
print(f"Nombre de lignes : {len(df)}")

scaler_cpu     = MinMaxScaler()
scaler_ram     = MinMaxScaler()
scaler_heure   = MinMaxScaler()
scaler_jour    = MinMaxScaler()
scaler_vitesse = MinMaxScaler()

cpu_norm     = scaler_cpu.fit_transform(df[["cpu_percent"]].values)
ram_norm     = scaler_ram.fit_transform(df[["ram_percent"]].values)
heure_norm   = scaler_heure.fit_transform(df[["heure"]].values)
jour_norm    = scaler_jour.fit_transform(df[["jour_semaine"]].values)
weekend_norm = df[["est_weekend"]].values
vitesse_norm = scaler_vitesse.fit_transform(df[["vitesse_cpu"]].values)

data_normalise = np.hstack([cpu_norm, ram_norm, heure_norm, jour_norm, weekend_norm, vitesse_norm])

FENETRE = 24
X, y = [], []
for i in range(len(data_normalise) - FENETRE - 24):
    X.append(data_normalise[i:i+FENETRE])
    y.append(np.hstack([cpu_norm[i+FENETRE:i+FENETRE+24], ram_norm[i+FENETRE:i+FENETRE+24]]))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

split = int(len(X) * 0.8)

X_train = torch.tensor(X[:split]).to(device)
X_test  = torch.tensor(X[split:]).to(device)
y_train = torch.tensor(y[:split]).to(device)
y_test  = torch.tensor(y[split:]).to(device)

print(f"Train : {len(X_train)} | Test : {len(X_test)}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=6, hidden_size=64, batch_first=True)
        self.dense = nn.Linear(64, 48)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1, :])
        return out.view(-1, 24, 2)

model = LSTMModel().to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.MSELoss()

train_losses = []
test_losses  = []

best_loss = float('inf')
patience  = 15
counter   = 0

print("\nEntrainement en cours...")
for epoch in range(200):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test)
        test_loss = loss_fn(pred_test, y_test).item()

    train_losses.append(epoch_loss / len(train_loader))
    test_losses.append(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\lstm_model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping a epoch {epoch+1}")
            break

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/200 | Train Loss: {train_losses[-1]:.5f} | Test Loss: {test_loss:.5f}")

model.load_state_dict(torch.load(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\lstm_model.pth"))
model.eval()
print("Meilleur modele charge !")

with torch.no_grad():
    predictions_norm = model(X_test).cpu().numpy()

y_test_numpy = y_test.cpu().numpy()

predictions_cpu = np.array([scaler_cpu.inverse_transform(p[:, 0].reshape(-1,1)).flatten() for p in predictions_norm])
predictions_ram = np.array([scaler_ram.inverse_transform(p[:, 1].reshape(-1,1)).flatten() for p in predictions_norm])

y_test_reel_cpu = np.array([scaler_cpu.inverse_transform(y[:, 0].reshape(-1,1)).flatten() for y in y_test_numpy])
y_test_reel_ram = np.array([scaler_ram.inverse_transform(y[:, 1].reshape(-1,1)).flatten() for y in y_test_numpy])

plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(y_test_reel_cpu.flatten()[:200], label="CPU Reel", color="blue")
plt.plot(predictions_cpu.flatten()[:200], label="CPU Predit", color="orange", linestyle="--")
plt.title("Prediction CPU et RAM - LSTM (200 premieres heures du test)")
plt.ylabel("CPU %")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(y_test_reel_ram.flatten()[:200], label="RAM Reel", color="green")
plt.plot(predictions_ram.flatten()[:200], label="RAM Predit", color="red", linestyle="--")
plt.xlabel("Heures")
plt.ylabel("RAM %")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\prediction_globale.png")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Erreur Train")
plt.plot(test_losses,  label="Erreur Test")
plt.title("Evolution de l'erreur pendant l'entrainement")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\courbe_loss.png")
plt.show()

idx = 50
entree_cpu_reel = scaler_cpu.inverse_transform(X_test[idx].cpu().numpy()[:, 0].reshape(-1,1)).flatten()
entree_ram_reel = scaler_ram.inverse_transform(X_test[idx].cpu().numpy()[:, 1].reshape(-1,1)).flatten()

plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(range(24), entree_cpu_reel, label="CPU Reel (passe)", color="blue")
plt.plot(range(24, 48), y_test_reel_cpu[idx], label="CPU Reel (futur)", color="cyan")
plt.plot(range(24, 48), predictions_cpu[idx], label="CPU Predit", color="orange", linestyle="--")
plt.axvline(x=24, color="gray", linestyle=":", label="Maintenant")
plt.ylabel("CPU %")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(24), entree_ram_reel, label="RAM Reel (passe)", color="green")
plt.plot(range(24, 48), y_test_reel_ram[idx], label="RAM Reel (futur)", color="lime")
plt.plot(range(24, 48), predictions_ram[idx], label="RAM Predit", color="red", linestyle="--")
plt.axvline(x=24, color="gray", linestyle=":", label="Maintenant")
plt.xlabel("Heures")
plt.ylabel("RAM %")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\prediction_24h_jointes.png")
plt.show()

SEUIL_PIC   = 75
SEUIL_CREUX = 35

pics_cpu  = np.where(predictions_cpu > SEUIL_PIC)[0]
creux_cpu = np.where(predictions_cpu < SEUIL_CREUX)[0]
pics_ram  = np.where(predictions_ram > SEUIL_PIC)[0]
creux_ram = np.where(predictions_ram < SEUIL_CREUX)[0]

print(f"\n--- Analyse CPU ---")
print(f"Pics detectes (CPU > {SEUIL_PIC}%) : {len(pics_cpu)} heures")
print(f"Creux detectes (CPU < {SEUIL_CREUX}%) : {len(creux_cpu)} heures")

print(f"\n--- Analyse RAM ---")
print(f"Pics detectes (RAM > {SEUIL_PIC}%) : {len(pics_ram)} heures")
print(f"Creux detectes (RAM < {SEUIL_CREUX}%) : {len(creux_ram)} heures")

def calcul_metriques(y_reel, y_pred):
    mae  = mean_absolute_error(y_reel, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    y_reel_safe = np.where(y_reel == 0, 1e-5, y_reel)
    mape = np.mean(np.abs((y_reel - y_pred) / y_reel_safe)) * 100
    r2   = r2_score(y_reel, y_pred)
    return mae, rmse, mape, r2

mae_cpu, rmse_cpu, mape_cpu, r2_cpu = calcul_metriques(y_test_reel_cpu, predictions_cpu)
mae_ram, rmse_ram, mape_ram, r2_ram = calcul_metriques(y_test_reel_ram, predictions_ram)

print("\n" + "="*20 + " RAPPORT DE PERFORMANCE GLOBAL " + "="*20)
metrics_df = pd.DataFrame({
    "Metrique": ["MAE", "RMSE", "MAPE", "R2"],
    "CPU": [f"{mae_cpu:.2f}%", f"{rmse_cpu:.2f}%", f"{mape_cpu:.2f}%", f"{r2_cpu:.4f}"],
    "RAM": [f"{mae_ram:.2f}%", f"{rmse_ram:.2f}%", f"{mape_ram:.2f}%", f"{r2_ram:.4f}"]
})
print(metrics_df.to_string(index=False))
print("="*71)