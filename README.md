```markdown
# 🚀 Projet 11 — LunarLander-v3 RL Agent Deployment

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-38A169?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v3-blue?style=for-the-badge)](https://gymnasium.farama.org/)

Application d'**Apprentissage par Renforcement (RL)** basée sur l'environnement **LunarLander-v3** de Gymnasium. Ce projet met en œuvre une **architecture microservices découplée** isolant le moteur d'inférence/simulation (**FastAPI**) du tableau de bord de visualisation et d'analyse (**Streamlit & Plotly**).

---

## 🛠️ Architecture du Projet

```text
Projet_11/
├── app.py                            # Backend API (FastAPI) pour l'inférence et les simulations
├── gui.py                            # Frontend GUI (Streamlit & Plotly) pour l'animation et le Dashboard
├── models/
│   └── dqn_exp12_750k_final.zip      # Modèle DQN optimal entraîné (Stable-Baselines3)
├── logs/                             # Fichiers d'événements et logs de suivi TensorBoard
├── .env                              # Configuration d'environnement locale (API_URL)
├── .gitignore                        # Fichier d'exclusion pour Git
├── requirements.txt                  # Dépendances de déploiement Cloud (Render / Streamlit Cloud)
└── README.md                         # Documentation du projet

```

---

## 🔬 Benchmark Algorithmique : DQN vs PPO

Une phase d'analyse expérimentale a permis de comparer **DQN** (*Off-Policy*) et **PPO** (*On-Policy*) sur l'espace d'actions discrets de LunarLander-v3 :

| Critère / Algorithme | 🤖 DQN (Deep Q-Network) | ⚡ PPO (Proximal Policy Optimization) |
| --- | --- | --- |
| **Type de Politique** | *Off-Policy* (Replay Buffer) | *On-Policy* (Actor-Critic) |
| **Efficacité d'Échantillonnage** | **Excellente** (Réutilisation des transitions) | Modérée (Nécessite plus d'interactions) |
| **Stabilité sur Actions Discrètes** | **Trè haute** (Convergence vers score > 200) | Progression régulière mais variance élevée |
| **Résultat d'Expérimentation** | **Optimal (Expérience 12)** | Performance inférieure sur 750k timesteps |

> **Résultat Cible (Expérience 12)** : **232.90 ± 32.85** à 750 000 timesteps *(Problème résolu et hautement stable)*.

---

## ⚙️ Prérequis & Installation

### 1. Installation des dépendances

Pour installer l'environnement de développement localement avec **`uv`** (ou `pip`) :

```bash
# Avec uv (recommandé)
uv add fastapi uvicorn gymnasium stable-baselines3 opencv-python-headless imageio imageio-ffmpeg numpy pandas plotly streamlit python-dotenv tensorboard

# Ou avec pip
pip install -r requirements.txt

```

### 2. Variables d'environnement (`.env`)

Créez un fichier `.env` à la racine de votre projet :

```env
API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

```

---

## 📈 Suivi de l'Entraînement avec TensorBoard

Pendant la phase d'apprentissage, l'historique et les métriques de convergence sont sauvegardés dans `./logs/`.

Pour consulter le tableau de bord de suivi :

```bash
tensorboard --logdir=./logs/

```

* **URL locale** : [http://localhost:6006](http://localhost:6006)

---

## 🚀 Lancement & Exécution Locale

Le projet s'exécute en lançant simultanément l'API (Backend) et l'interface graphique (Frontend) dans deux terminaux distincts :

### 1. Démarrer le Backend FastAPI

```bash
uv run uvicorn app:app --reload

```

* **Documentation OpenAPI (Swagger)** : [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)
* **Endpoints HTTP principaux** :
* `GET /` : Vérification de la santé du serveur.
* `POST /predict` : Prenez un état de 8 variables continues et renvoie l'action optimale ($0, 1, 2, 3$).
* `GET /play` : Simule un épisode complet et retourne la trajectoire et l'animation MP4 en Base64.
* `GET /play_compare` : Simule deux épisodes sous la **MÊME seed** (*Sans Vent* vs *Avec Vent*) pour comparer la résistance du modèle.



### 2. Démarrer le Frontend Streamlit

```bash
uv run streamlit run gui.py

```

* **Accès web local** : [http://localhost:8501](http://localhost:8501)

---

## 📊 Fonctionnalités du Dashboard Streamlit

1. **Relecture Vidéo MP4 & Player HTML5** :
* Génération et récréation fluide de la trajectoire d'atterrissage.


2. **Dashboard Analytique Plotly** :
* **KPIs Globaux** : Score total, nombre de pas, statut d'atterrissage.
* **Distribution des Actions (Plotly Pie)** : Proportion des moteurs allumés.
* **Évolution du Score Cumulé (Plotly Line)** : Progression des gains au fil du vol.
* **Profil de Vol 2D (Plotly Scatter)** : Trajectoire spatiale ($X$ vs $Y$) colorée selon l'action décidée par le réseau de neurones.


3. **Mode Comparatif Côte à Côte (`/play_compare`)** :
* Comparaison sous **seed identique** de la résilience du pilotage avec et sans turbulences de vent.



---

## 🌐 Déploiement Cloud

* **Backend API (FastAPI)** : Déployé sur **Render** (`https://<votre-api>.onrender.com`)
* **Frontend GUI (Streamlit)** : Déployé sur **Streamlit Community Cloud** (avec Secret `API_URL` configuré).

```

---
