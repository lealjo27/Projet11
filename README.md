lancer fastapi :  uv run uvicorn app:app --reload
Lancer Streamlit : uv run streamlit run gui.py
Lancer tensorboard : tensorboard --logdir ./dqn_lunar_tensorboard/


Pour rédiger le fichier `README.md` complet et structuré de ton projet (Projet 11 - LunarLander-v3), voici le contenu exact au format Markdown.

Crée le fichier `README.md` à la racine de ton dossier `/home/jo/Documents/uv/Projet_11/` et insère le texte ci-dessous :

```markdown
# 🚀 Project 11 - LunarLander-v3 RL Agent Deployment

Ce projet propose une application d'**Apprentissage par Renforcement (RL)** basée sur l'environnement **LunarLander-v3** de Gymnasium[cite: 1, 2]. 
Il met en œuvre une architecture découplée isolant le moteur d'inférence/simulation (**FastAPI**) du frontend de visualisation et d'analyse (**Streamlit & Plotly**)[cite: 1, 2].

---

## 🛠️ Architecture du Projet

```text
Projet_11/
├── app.py                            # Backend API (FastAPI) pour l'inférence et la simulation[cite: 1, 2]
├── gui.py                            # Frontend GUI (Streamlit & Plotly) pour l'animation et le Dashboard[cite: 1, 2]
├── models/
│   └── dqn_exp12_750k_final.zip      # Modèle DQN entraîné (Stable-Baselines3)[cite: 1]
├── .env                              # Variables d'environnement locales (API_URL)[cite: 1, 2]
├── .gitignore                        # Fichier d'exclusion pour Git[cite: 1, 2]
├── requirements.txt                  # Dépendances de déploiement Cloud (Render / Streamlit Cloud)[cite: 1, 2]
└── README.md                         # Documentation du projet[cite: 1, 2]

```

---

## ⚙️ Prérequis & Installation

### 1. Installation des dépendances

Pour installer les dépendances du projet localement avec `uv` (ou `pip` dans votre environnement virtuel) :

```bash
# Avec uv
uv add fastapi uvicorn gymnasium stable-baselines3 opencv-python-headless numpy pandas plotly streamlit python-dotenv

# Ou avec pip
pip install -r requirements.txt

```

### 2. Variables d'environnement (.env)

Créez un fichier `.env` à la racine de votre projet :

```env
API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

```

---

## 🚀 Utilisation & Execution Locale

Le système nécessite l'exécution séparée du backend et du frontend dans **deux terminaux**.

### 1. Démarrer l'API FastAPI Backend

Dans le premier terminal :

```bash
uv run uvicorn app:app --reload

```

* **Interface OpenAPI (Swagger UI)** : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


* **Endpoints HTTP principaux** :
* `GET /` : Vérification du statut de l'API.


* `POST /predict` : Reçoit un état de 8 variables et renvoie l'action déterministe ($0, 1, 2, 3$).


* `GET /play` : Simule un épisode complet côté serveur et retourne la trajectoire ainsi que l'animation MP4 (Base64).





### 2. Démarrer l'Interface Streamlit Frontend

Dans le second terminal :

```bash
uv run streamlit run gui.py

```

* **Accès web local** : [http://localhost:8501](http://localhost:8501)



---

## 📊 Fonctionnalités du Tableau de Bord (Streamlit)

1. **Relecture Vidéo en Direct** :


* Déclenchement des simulations via l'endpoint `GET /play` de l'API FastAPI.


* Lecteur vidéo HTML5 natif pour une relecture fluide de l'atterrissage.




2. **Dashboard Analytique Interactive (Plotly)** :


* **KPIs Globaux** : Score total, durée (pas), statut (succès/échec), score moyen par pas.


* **Distribution des Actions (Plotly Pie)** : Proportions des décisions (Inaction, Moteur Principal, Moteurs Latéraux).


* **Score Cumulé (Plotly Line)** : Évolution de la récompense cumulée au fil des pas de temps.


* **Profil de Vol 2D (Plotly Scatter)** : Trajectoire spatiale ($Position X$ vs $Altitude Y$) colorée par action avec zone d'atterrissage repère.


* **Circonstances de Vol** : Histogrammes et barres d'erreurs (moyennes et écarts-types) de la récompense selon les paliers d'altitude et les vitesses verticales.





---

## 🌐 Déploiement Cloud

* **Backend FastAPI** : Déployé sur **Render** (`https://<votre-api>.onrender.com`).


* **Frontend Streamlit** : Déployé sur **Streamlit Community Cloud** avec la variable de secret TOML `API_URL` configurée pour pointer vers l'API Render.



```

```