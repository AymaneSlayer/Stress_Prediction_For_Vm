Ce projet utilise un réseau de neurones récurrents (LSTM) pour anticiper la charge d'une infrastructure système (CPU et RAM) 24 heures à l'avance, permettant de prévenir les saturations de ressources.

## Fonctionnalités
- Entraînement d'un modèle LSTM Multi-Input & Multi-Output (analyse conjointe et simultanée du CPU et de la RAM).
- Architecture prédictive basée sur des fenêtres glissantes de 24 heures.
- Système d'alerte visuelle en cas de dépassement de seuil critique (>75%).
- Interface web interactive développée avec **Streamlit** pour visualiser l'historique récent et les courbes de prévisions futures.

## Structure du Projet
- `Prédiction.py` : Script d'entraînement du modèle, gestion de la normalisation, de l'Early Stopping et génération des graphiques d'évaluation.
- `app.py` : Code de l'application web Streamlit (chargement du modèle sur CPU, mise en cache des ressources et interface utilisateur).
- `requirements.txt` : Liste des dépendances Python requises pour exécuter le projet.
- `.gitignore` : Configuration Git pour exclure les fichiers de données lourds (.csv) et les poids du modèle (.pth).

## Installation et Lancement

1. **Cloner le dépôt localement :**
   ```bash
   git clone [https://github.com/AymaneSlayer/Stress_Prediction_For_Vm.git](https://github.com/AymaneSlayer/Stress_Prediction_For_Vm.git)
   cd Stress_Prediction_For_Vm
2. Installer les bibliothèques requises
3. Fichiers locaux requis (non inclus sur GitHub) :
Pour faire tourner le projet en local, place ton fichier de données Book1.csv ainsi que ton modèle entraîné lstm_model.pth à la racine du dossier
4. Lancer l'interface graphique :
streamlit run app.py
