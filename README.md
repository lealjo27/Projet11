```markdown
# 🚀 Project 11 - LunarLander-v3 RL Agent (DQN vs PPO, Benchmark & Deployment)

Ce projet propose une solution complète d'**Apprentissage par Renforcement (RL)** sur l'environnement **LunarLander-v3** de Gymnasium[cite: 1, 3, 4]. 

Il couvre la totalité du cycle de développement :
1. **Études et Benchmark Algorithmique** : Comparaison approfondie entre **DQN** (Deep Q-Network) et **PPO** (Proximal Policy Optimization)[cite: 1, 3].
2. **Monitoring & Logs** : Suivi fin de la convergence de l'apprentissage via **TensorBoard**[cite: 1, 2].
3. **Architecture Découplée (Production)** :
   * Backend Inférence & Simulation (**FastAPI**)[cite: 1, 3, 4].
   * Frontend Interactif & Tableau de Bord (**Streamlit & Plotly**)[cite: 1, 3].

---

## 🔬 Benchmark Algorithmique : DQN vs PPO

Pour résoudre le problème d'atterrissage sur la lune, deux grandes familles d'algorithmes de RL continu/discret ont été comparées sur **LunarLander-v3**[cite: 1, 3, 4] :

### 1. DQN (Deep Q-Network) — *Off-Policy*
* **Principe** : Algorithme basé sur la fonction de valeur $Q(s, a)$ avec l'utilisation d'un *Replay Buffer* pour casser la corrélation temporelle des données et d'un *Target Network* pour stabiliser les cibles de Bellman[cite: 1].
* **Constat & Résultats** :
  * DQN s'est révélé extrêmement performant sur les espaces d'action discrets de LunarLander (4 actions discrets)[cite: 3, 4].
  * **Expérience 12 (750k timesteps)** : C'est la configuration optimale retenue pour le déploiement[cite: 3, 4]. Elle atteint un score moyen de **232.90 ± 32.85** (considéré comme résolu et hautement stable)[cite: 3].

### 2. PPO (Proximal Policy Optimization) — *On-Policy*
* **Principe** : Algorithme *Actor-Critic* optimisant directement la politique via un découpage du gradient (*clipping ratio*) pour éviter des mises à jour trop violentes de la politique.
* **Analyse Comparative avec DQN** :
  * **Efficacité d'échantillonnage (*Sample Efficiency*)** : DQN a montré une meilleure efficacité sur les données grâce à la réutilisation des transitions passées stockées dans le *Replay Buffer*, alors que PPO nécessite plus d'interactions fraîches avec l'environnement (*On-Policy*).
  * **Stabilité & Variance** : Bien que PPO offre une montée en score plus régulière au début, DQN (avec une phase d'exploration bien réglée et un volume de 750k pas) a convergé vers une politique d'atterrissage nettement plus précise avec un score final moyen plus élevé sur cette tâche spécifique[cite: 3].

---

## 🛠️ Architecture du Projet

```text
Projet_11/
├── logs/                             # Enregistrements TensorBoard (fichiers d'événements SB3)
├── models/
│   └── dqn_exp12_750k_final.zip      # Modèle DQN optimal entraîné à 750k timesteps
├── app.py                            # FastAPI Backend (Inférence, simulation & comparaison)
├── gui.py                            # Streamlit Frontend (Relecture MP4, Dashboard Plotly & Visualisation)
├── .env                              # Variables d'environnement locales (API_URL)
├── .gitignore                        # Configuration d'exclusion Git
├── requirements.txt                  # Dépendances pour Render & Streamlit Cloud
└── README.md                         # Documentation complète du projet

```

---

## ⚙️ Prérequis & Installation

### 1. Installation des dépendances

Pour installer les paquets requis avec `uv` (ou `pip` dans votre environnement virtuel) :

```bash
# Avec uv
uv add fastapi uvicorn gymnasium stable-baselines3 tensorboard opencv-python-headless imageio imageio-ffmpeg numpy pandas plotly streamlit python-dotenv

# Avec pip
pip install -r requirements.txt

```

### 2. Configuration d'environnement (.env)

Créez un fichier `.env` à la racine de votre projet :

```env
API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

```

---

## 📈 Monitoring & Analyse de l'Entraînement avec TensorBoard

Pendant les phases d'expérimentation et d'évaluation comparative (DQN vs PPO), les métriques ont été journalisées dans le dossier `./logs/`.

### Démarrage de TensorBoard

```bash
tensorboard --logdir=./logs/

```

* **URL d'accès locale** : [http://localhost:6006/](http://localhost:6006/)


* **Métriques analysées** :
* `rollout/ep_rew_mean` : Récompense moyenne accumulée (passage du seuil de succès > 200).


* `rollout/ep_len_mean` : Durée moyenne des épisodes (efficacité du vol).
* `train/loss` : Perte de la fonction de valeur et stabilisation du réseau.



---

## 🚀 Utilisation & Exécution Locale

L'application repose sur une exécution dissociée du backend et du frontend dans deux terminaux séparés.

### 1. Démarrer l'API Backend (FastAPI)

```bash
uv run uvicorn app:app --reload

```

* **Interface Swagger (Docs API)** : [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)


* **Endpoints HTTP principaux** :
* `GET /` : Endpoint de contrôle de santé (*Healthcheck*).


* `POST /predict` : Prend les 8 variables d'état et retourne l'action déterministe ($0, 1, 2, 3$).


* `GET /play` : Simule un épisode complet, renvoie les données de trajectoire et l'animation MP4 encodée en Base64.


* `GET /play_compare` : Exécute deux simulations simultanées sous la **MÊME seed** (*Sans Vent* vs *Avec Vent*) pour étudier la robustesse du modèle face aux turbulences.





### 2. Démarrer l'Interface Frontend (Streamlit)

```bash
uv run streamlit run gui.py

```

* **Accès web local** : [http://localhost:8501](http://localhost:8501)



---

## 📊 Tableau de Bord Interactif (Streamlit & Plotly)

1. **Analyse de Vol Unique (`/play`)** :


* Relecture vidéo fluide de l'atterrissage via lecteur HTML5 natif.


* Profil 2D de la trajectoire (Position X vs Altitude Y), camembert de distribution des réacteurs allumés, et évolution des scores cumulés.




2. **Mode Comparatif de Perturbation (`/play_compare`)** :


* Analyse côte à côte avec des conditions initiales strictement identiques (*Seed fixée*).


* Évaluation de la résistance du pilotage DQN face au vent (comparaison des trajectoires, temps de vol et pénalités de carburant).





---

## 🌐 Déploiement Cloud

* **Backend FastAPI** : Hébergé sur **Render** (`https://<votre-api>.onrender.com`).


* **Frontend Streamlit** : Hébergé sur **Streamlit Community Cloud** avec la variable d'environnement/secret TOML `API_URL` configurée pour communiquer avec l'API sur Render.



```
