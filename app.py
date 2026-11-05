from contextlib import asynccontextmanager
import numpy as np
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from stable_baselines3 import DQN
import base64
import cv2  # opencv-python

import os, tempfile
import imageio

# Variable globale pour stocker le modèle en mémoire
model = None

# Gestionnaire du cycle de vie (lifespan) recommandé par FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = DQN.load("models/dqn_exp12_750k_final.zip")
        print("Modèle LunarLander chargé avec succès.")
    except Exception as e:
        print(f"Erreur de chargement du modèle: {e}")
    yield
    # Nettoyage si nécessaire à la fermeture de l'API

# 1. Initialisation de FastAPI avec le lifespan
app = FastAPI(
    title="LunarLander RL API",
    description="API Backend pour interagir avec le modèle DQN entraîné sur LunarLander-v3",
    version="1.0",
    lifespan=lifespan
)
# Pydantic : validation des données d'entrée.
class StateInput(BaseModel):
    """Modèle de données pour l'état de l'environnement"""
    state: List[float] = Field(...,
                               example=[0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                               description="État de l'environnement LunarLander-v3 (8 dimensions)")

    
@app.get("/")
def home():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    return {"status": "ok", "message": "API LunarLander-v3 active"}

@app.post("/predict")
def predict_action(input_data: StateInput):
    """Reçoit un état à 8 variables et renvoie l'action choisie par l'agent"""
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non disponible.")
    if len(input_data.state) != 8:
        raise HTTPException(status_code=400, detail="L'état doit contenir exactement 8 valeurs. ( reçu : {len(input_data.state)} )")

    obs = np.array(input_data.state, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return {"action": int(action)}


@app.get("/play")
def play_episode():
    """Simule une partie complète et renvoie la trajectoire avec les images du jeu."""
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle RL non disponible.")
    
    # On spécifie render_mode="rgb_array" pour récupérer les images du jeu
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, info = env.reset()
    
    trajectory = []
    frames = []
        
    total_reward = 0.0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Capture de l'image de la frame courante
        frames.append(env.render())
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        trajectory.append({
            "state": obs.tolist(),
            "action": int(action),
            "reward": float(reward),
            })
        
        total_reward += float(reward)
        obs = next_obs
        done = terminated or truncated
        
    env.close()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "episode.mp4")
        imageio.mimsave(path, frames, fps=30, codec="libx264",pixelformat="yuv420p")
        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    return {
        "total_reward": total_reward,
        "steps": len(trajectory),
        "trajectory": trajectory,
        "video": video_b64
    }