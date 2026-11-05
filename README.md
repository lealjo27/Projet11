<h1 align="center">🚀 Projet 11 — LunarLander-v3 RL Agent</h1>
<p align="center"><b>DQN vs PPO · Benchmark & Deployment</b></p>

<p align="center">
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://stable-baselines3.readthedocs.io/"><img src="https://img.shields.io/badge/Stable--Baselines3-38A169?style=for-the-badge" alt="Stable-Baselines3"></a>
  <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-v3-blue?style=for-the-badge" alt="Gymnasium"></a>
</p>

Application d'**Apprentissage par Renforcement (RL)** basée sur l'environnement **LunarLander-v3** de Gymnasium. Ce projet met en œuvre une **architecture microservices découplée** isolant le moteur d'inférence/simulation (**FastAPI**) du tableau de bord interactif de visualisation et d'analyse (**Streamlit & Plotly**).

---

## 📌 Sommaire

- [🔬 Benchmark Algorithmique](#-benchmark-algorithmique--dqn-vs-ppo)
- [🧪 Résultats des Expériences](#-résultats-des-expériences)
- [🛠️ Architecture du Projet](#️-architecture-du-projet)
- [⚙️ Prérequis & Installation](#️-prérequis--installation)
- [📈 Monitoring & Logs (TensorBoard)](#-monitoring--logs-tensorboard)
- [🚀 Lancement & Exécution Locale](#-lancement--exécution-locale)
- [📊 Tableau de Bord Interactif](#-tableau-de-bord-interactif)

---

## 🔬 Benchmark Algorithmique : DQN vs PPO

Une phase d'analyse expérimentale approfondie a permis de comparer deux grandes familles d'algorithmes de RL sur l'espace d'actions discrets de **LunarLander-v3** :

| Critère / Algorithme | 🤖 DQN (Deep Q-Network) | ⚡ PPO (Proximal Policy Optimization) |
| :--- | :--- | :--- |
| **Efficacité d'Échantillonnage** | **Excellente** (Réutilisation des transitions) | **Modérée** (Nécessite plus d'interactions) |
| **Stabilité sur Actions Discrètes** | **Très haute** (Score moyen > 200) | Progression régulière mais variance élevée |
| **Résultat d'Expérimentation** | **Optimal (Expérience 12)** | Performance inférieure sur 750k timesteps |

> 🎯 **Résultat Retenu (Expérience 12)** : **232.90 ± 32.85** à 750 000 timesteps
> *(Environnement résolu et hautement stable)*

---

## 🧪 Résultats des Expériences

Chaque expérience modifie **un seul hyperparamètre** par rapport à la précédente, pour isoler son impact. Les scores sont calculés sur **100 épisodes** d'évaluation.

| Expérience | Hyperparamètre modifié | Valeur | Paramètres inchangés | Objectif visé | Score moyen | Écart-type (σ) | Statut |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Exp 0** (Baseline) | Référence initiale | 20 000 steps | Défaut SB3 | Établir le point de départ | -184.48 | 51.00 | ✅ Terminé |
| **Exp 1** (Volume) | `total_timesteps` | **300 000** | Défaut SB3 | Isoler l'impact de la durée d'entraînement | -46.93 | 34.60 | ✅ Terminé |
| **Exp 2** (Apprentissage) | `learning_rate` | **`1e-4`** | 300k steps, reste par défaut | Stabiliser la mise à jour des poids | -38.58 | 52.34 | ✅ Terminé |
| **Exp 3** (Exploration) | `exploration_fraction` | **`0.2`** | 300k steps, `lr=1e-4` | Étendre la phase d'exploration ε-greedy | 7.10 | 107.42 | ✅ Terminé |
| **Exp 4** (Mémoire) | `buffer_size` | **`100 000`** | 300k steps, `lr=1e-4`, `exp_frac=0.2` | Diversifier le Replay Buffer | -19.14 | 94.40 | ✅ Terminé |
| **Exp 5** (Démarrage) | `learning_starts` | **`10 000`** | Config Exp 4 | Accumuler des expériences variées avant l'apprentissage | -19.60 | 65.58 | ✅ Terminé |
| **Exp 6** (Architecture) | `policy_kwargs` | **`net_arch=[128, 128]`** | Config Exp 5 | Augmenter la capacité du réseau | 39.68 | 118.37 | ✅ Terminé |
| **Exp 7** (Cible) | `target_update_interval` | **`1000`** | Config Exp 6 | Stabiliser la cible d'apprentissage | 30.54 | 73.22 | ✅ Terminé |
| **Exp 8** (Remise) | `gamma` | **`0.999`** | Config Exp 7 | Tester un horizon temporel long | -45.69 | 22.06 | ✅ Terminé |
| **Exp 9** (Lot) | `batch_size` | **`128`** | Config Exp 7 (`gamma=0.99`) | Réduire le bruit du gradient | 58.17 | 96.10 | ✅ Terminé |
| **Exp 10** (Consolidation) | `total_timesteps` | **`500 000`** | Config Exp 9 | Consolider la convergence | **234.87** | **43.72** | 🏆 Résolu |
| **Exp 11** (Extension 1M) | `total_timesteps` | **`1 000 000`** | Config Exp 10 | Évaluer la stabilité à très long terme | **189.81** | **62.98** |  Terminé |
| **Exp 12** (Volume 750k) | `total_timesteps` | **`750 000`** | Config Exp 10 | Ajuster durée de convergence et régularité | **232.90** | **32.85** | 🥇 **Optimal** |
| **Exp 13** (Extension 800k) | `total_timesteps` | **`800 000`** | Config Exp 12 | Tester l'impact de 50k steps supplémentaires | **220.08** | **34.99** | Terminé |

### 🔎 Analyse

- **Volume d'entraînement** : passer de 20k à 300k steps fait gagner environ 138 points (Exp 0 → 1).
- **Réseau, lot et durée** : `net_arch=[128, 128]` (Exp 6), `batch_size=128` (Exp 9) et 500k steps (Exp 10) font franchir le seuil de 200.
- **`gamma=0.999` dégrade le score** (Exp 8) : l'horizon trop long est écarté.
- **Pas de gain au-delà de 750k** : à 800k (Exp 13) et à 1M (Exp 11), le score baisse.
- **Choix final : Exp 12.** Son score moyen est proche de l'Exp 10 (232.90 contre 234.87), mais il a l'**écart-type le plus faible** (32.85), donc l'atterrissage est le plus régulier.


---

## 🛠️ Architecture du Projet

```text
Projet_11/
├── app.py                         # Backend API (FastAPI) : inférence, simulation, comparaison
├── gui.py                         # Frontend GUI (Streamlit & Plotly) : relecture vidéo et Dashboard
├── models/
│   └── dqn_exp12_750k_final.zip   # Modèle DQN optimal entraîné (Stable-Baselines3)
├── logs/                          # Fichiers d'événements et logs de suivi TensorBoard
├── .env                           # Configuration d'environnement locale (API_URL)
├── .gitignore                     # Configuration d'exclusion Git
├── requirements.txt               # Dépendances de déploiement Cloud (Render / Streamlit Cloud)
└── README.md                      # Documentation complète du projet
```

---

## ⚙️ Prérequis & Installation

### 1. Installation des dépendances

Installez l'environnement de développement localement avec **`uv`** (ou `pip`) :

```bash
# Avec uv (recommandé)
uv add fastapi uvicorn gymnasium stable-baselines3 opencv-python-headless imageio imageio-ffmpeg numpy pandas plotly streamlit python-dotenv tensorboard

# Ou avec pip
pip install -r requirements.txt
```

### 2. Configuration d'environnement (`.env`)

Créez un fichier `.env` à la racine de votre projet :

```env
API_URL=http://127.0.0.1:8000
```

---

## 📈 Monitoring & Logs (TensorBoard)

Pendant la phase d'apprentissage avec Stable-Baselines3, l'historique et les métriques de convergence ont été enregistrés dans `./logs/`.

Pour consulter le tableau de bord de suivi :

```bash
tensorboard --logdir=./logs/
```

🔗 **URL locale** : <http://localhost:6006>

---

## 🚀 Lancement & Exécution Locale

Le système s'exécute en séparant le backend et le frontend dans **deux terminaux distincts**.

### 1. Démarrer le Backend FastAPI

```bash
uv run uvicorn app:app --reload
```

📖 **Documentation OpenAPI (Swagger)** : <http://127.0.0.1:8000/docs>

**Endpoints HTTP principaux**

| Méthode | Route | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Endpoint de contrôle de santé (*Healthcheck*) |
| `POST` | `/predict` | Reçoit un état de 8 variables continues et renvoie l'action déterministe (`0`, `1`, `2`, `3`) |
| `GET` | `/play` | Simule un épisode complet et retourne la trajectoire ainsi que l'animation MP4 en Base64 |
| `GET` | `/play_compare` | Simule deux parties sous la **MÊME seed** (*Sans Vent* vs *Avec Vent*) pour évaluer la résistance du modèle aux turbulences |

### 2. Démarrer le Frontend Streamlit

```bash
uv run streamlit run gui.py
```

🔗 **Accès web local** : <http://localhost:8501>

---

## 📊 Tableau de Bord Interactif

### 1. Analyse de Vol Unique (`/play`)

- Relecture vidéo fluide de l'atterrissage via le lecteur HTML5 natif.
- Graphiques Plotly :
  - Camembert de distribution des réacteurs (Pie)
  - Profil de vol 2D (Scatter X vs Y)
  - Évolution du score cumulé (Line)

### 2. Mode Comparatif Côte à Côte (`/play_compare`)

- Comparaison directe sous **conditions initiales strictement identiques (Seed fixée)**.
- Double vidéo et comparaison parallèle des KPIs (Score final, Pas de temps, Statut d'atterrissage).

---

## 👤 Auteur

**Jo — [@lealjo27](https://github.com/lealjo27)**

Projet réalisé dans le cadre de la formation **AI Engineer OpenClassrooms**.
