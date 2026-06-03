import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilise : {device}")

# 1. Lecture du csv et nettoyage
df = pd.read_csv(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\Book1.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
print(f"Nombre de lignes : {len(df)}")

# 2. Normalisation (1 scaler par feature)
scaler_cpu = MinMaxScaler()
scaler_ram = MinMaxScaler()

cpu_norm = scaler_cpu.fit_transform(df[["cpu_percent"]].values)
ram_norm = scaler_ram.fit_transform(df[["ram_percent"]].values)

# Combiner CPU + RAM → shape (n, 2)
data_normalise = np.hstack([cpu_norm, ram_norm])

FENETRE = 24

# X aura les 24h de CPU+RAM, y aura les 24h de CPU ET de RAM suivantes
X, y = [], []
for i in range(len(data_normalise) - FENETRE - 24):
    X.append(data_normalise[i:i+FENETRE])              # (24, 2) : CPU + RAM passés
    y.append(data_normalise[i+FENETRE:i+FENETRE+24])   # (24, 2) : CPU + RAM futurs (Multi-Output)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

split = int(len(X) * 0.8) # 80% entrainement 20% test

X_train = torch.tensor(X[:split]).to(device)
X_test  = torch.tensor(X[split:]).to(device)
y_train = torch.tensor(y[:split]).to(device)
y_test  = torch.tensor(y[split:]).to(device)

print(f"Train : {len(X_train)} | Test : {len(X_test)}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)

# 3. Modele LSTM (Mis a jour pour predire 2 variables sur 24 heures -> 48 sorties)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.dense = nn.Linear(64, 48) # 24h * 2 variables = 48

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1, :])      # Sortie de taille (batch_size, 48)
        return out.view(-1, 24, 2)           # Redimensionne en (batch_size, 24, 2)

model = LSTMModel().to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.MSELoss() # Erreur quadratique moyenne

train_losses = []
test_losses  = []

# Early Stopping
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

    # Erreur test
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test)
        test_loss = loss_fn(pred_test, y_test).item()

    train_losses.append(epoch_loss / len(train_loader))
    test_losses.append(test_loss)

    # Early stopping : sauvegarde le meilleur modele
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

# 4. Chargement du meilleur modele
model.load_state_dict(torch.load(r"C:\Users\MSI\Desktop\Aymane msi\ecc\Stage Inwi\lstm_model.pth"))
model.eval()
print("Meilleur modele charge !")

# 5. Prediction et Denormalisation
with torch.no_grad():
    predictions_norm = model(X_test).cpu().numpy()

y_test_numpy = y_test.cpu().numpy()

# Denormalisation separee pour le CPU et la RAM
predictions_cpu = np.array([scaler_cpu.inverse_transform(p[:, 0].reshape(-1,1)).flatten() for p in predictions_norm])
predictions_ram = np.array([scaler_ram.inverse_transform(p[:, 1].reshape(-1,1)).flatten() for p in predictions_norm])

y_test_reel_cpu = np.array([scaler_cpu.inverse_transform(y[:, 0].reshape(-1,1)).flatten() for y in y_test_numpy])
y_test_reel_ram = np.array([scaler_ram.inverse_transform(y[:, 1].reshape(-1,1)).flatten() for y in y_test_numpy])

# 6. Generation des Graphes

# Graphe 1 : Reel vs Predit (Visualisation sur les 200 premieres heures de test)
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

# Graphe 2 : Courbe de loss
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

# Graphe 3 : Focus sur une fenetre specifique (24h passees + 24h futures)
idx = 50
entree_cpu_reel = scaler_cpu.inverse_transform(X_test[idx].cpu().numpy()[:, 0].reshape(-1,1)).flatten()
entree_ram_reel = scaler_ram.inverse_transform(X_test[idx].cpu().numpy()[:, 1].reshape(-1,1)).flatten()

plt.figure(figsize=(14, 6))
# Sous-graphe pour le CPU
plt.subplot(2, 1, 1)
plt.plot(range(24), entree_cpu_reel, label="CPU Reel (passe)", color="blue")
plt.plot(range(24, 48), y_test_reel_cpu[idx], label="CPU Reel (futur)", color="cyan")
plt.plot(range(24, 48), predictions_cpu[idx], label="CPU Predit", color="orange", linestyle="--")
plt.axvline(x=24, color="gray", linestyle=":", label="Maintenant")
plt.ylabel("CPU %")
plt.legend()

# Sous-graphe pour la RAM
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

# 7. Analyse des Pics, Creux et Erreurs
SEUIL_PIC   = 75
SEUIL_CREUX = 35

pics_cpu  = np.where(predictions_cpu > SEUIL_PIC)[0]
creux_cpu = np.where(predictions_cpu < SEUIL_CREUX)[0]
pics_ram  = np.where(predictions_ram > SEUIL_PIC)[0]
creux_ram = np.where(predictions_ram < SEUIL_CREUX)[0]

print(f"\n--- Analyse CPU ---")
print(f"Pics detectes (CPU > {SEUIL_PIC}%) : {len(pics_cpu)} heures")
print(f"Creux detectes (CPU < {SEUIL_CREUX}%) : {len(creux_cpu)} heures")
mae_cpu = np.mean(np.abs(predictions_cpu - y_test_reel_cpu))
print(f"Erreur moyenne (MAE CPU) : {mae_cpu:.2f}%")

print(f"\n--- Analyse RAM ---")
print(f"Pics detectes (RAM > {SEUIL_PIC}%) : {len(pics_ram)} heures")
print(f"Creux detectes (RAM < {SEUIL_CREUX}%) : {len(creux_ram)} heures")
mae_ram = np.mean(np.abs(predictions_ram - y_test_reel_ram))
print(f"Erreur moyenne (MAE RAM) : {mae_ram:.2f}%")