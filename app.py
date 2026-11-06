import base64
from contextlib import asynccontextmanager
import gc
import os
import random
import tempfile
from typing import List, Optional

import cv2
import gymnasium as gym
import imageio
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import DQN

# Variable globale pour stocker le modèle en mémoire
model = None


# Gestionnaire du cycle de vie (lifespan) recommandé par FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
  global model
  try:
    model_path = "models/dqn_exp12_750k_final.zip"
    if os.path.exists(model_path):
      model = DQN.load(model_path)
      print("Modèle LunarLander chargé avec succès.")
    else:
      print(f"Attention : Modèle introuvable à l'emplacement '{model_path}'.")
  except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
  yield
  # Nettoyage explicite à la fermeture de l'API
  model = None
  gc.collect()


# Initialisation de FastAPI avec le lifespan
app = FastAPI(
    title="LunarLander RL API",
    description=(
        "API Backend pour l'inférence, la simulation et la comparaison sous"
        " Gymnasium"
    ),
    version="1.0",
    lifespan=lifespan,
)


# Pydantic : validation des données d'entrée.
class StateInput(BaseModel):
  """Modèle de données pour l'état de l'environnement"""

  state: List[float] = Field(
      ...,
      example=[0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      description="État continu de l'environnement LunarLander-v3 (8 variables)",
  )


@app.api_route("/", methods=["GET", "HEAD"])
def home():
  """Endpoint de santé pour vérifier que l'API fonctionne"""
  return {"status": "ok", "message": "API LunarLander-v3 active"}


@app.post("/predict")
def predict_action(input_data: StateInput):
  """Reçoit un état à 8 variables et renvoie l'action choisie par l'agent"""
  if model is None:
    raise HTTPException(status_code=500, detail="Modèle RL non disponible.")
  if len(input_data.state) != 8:
    raise HTTPException(
        status_code=400,
        detail=(
            "L'état doit contenir exactement 8 valeurs (reçu :"
            f" {len(input_data.state)})."
        ),
    )

  obs = np.array(input_data.state, dtype=np.float32)
  action, _ = model.predict(obs, deterministic=True)
  return {"action": int(action)}


@app.get("/play")
def play_episode(enable_wind: bool = False, wind_power: float = 15.0):
  """Exécute un épisode complet avec ou sans vent et renvoie la vidéo MP4 en Base64."""
  if model is None:
    raise HTTPException(status_code=500, detail="Modèle RL non disponible.")

  env_kwargs = {}
  if enable_wind:
    env_kwargs = {
        "enable_wind": True,
        "wind_power": wind_power,
    }

  env = gym.make("LunarLander-v3", render_mode="rgb_array", **env_kwargs)
  obs, info = env.reset()

  trajectory = []
  frames = []
  total_reward = 0.0
  done = False

  while not done:
    action, _ = model.predict(obs, deterministic=True)

    # Capture et redimensionnement (304x200) pour optimiser la mémoire RAM
    frame = env.render()
    frame_resized = cv2.resize(frame, (304, 200))
    frames.append(frame_resized)

    next_obs, reward, terminated, truncated, info = env.step(action)

    trajectory.append({
        "state": obs.tolist(),
        "action": int(action),
        "reward": float(reward),
    })

    total_reward += float(reward)
    obs = next_obs
    done = terminated or truncated

  # Fermeture immédiate et nettoyage de l'environnement
  env.close()
  del env

  # Encodage léger de la vidéo MP4
  video_b64 = ""
  if frames:
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "episode.mp4")
      imageio.mimsave(
          path,
          frames,
          fps=30,
          codec="libx264",
          pixelformat="yuv420p",
          macro_block_size=None,
      )
      with open(path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

  # Nettoyage explicite de la mémoire vive
  del frames
  gc.collect()

  return {
      "total_reward": total_reward,
      "steps": len(trajectory),
      "trajectory": trajectory,
      "video": video_b64,
  }


@app.get("/play_compare")
def play_compare(wind_power: float = 15.0, seed: Optional[int] = None):
  """Exécute deux simulations à partir de la MÊME graine (seed) :

  une partie SANS vent et une partie AVEC vent.
  """
  if model is None:
    raise HTTPException(status_code=500, detail="Modèle RL non disponible.")

  if seed is None:
    seed = random.randint(0, 10000)

  results = {}

  for wind_enabled in [False, True]:
    env_kwargs = {}
    if wind_enabled:
      env_kwargs = {
          "enable_wind": True,
          "wind_power": wind_power,
      }

    env = gym.make("LunarLander-v3", render_mode="rgb_array", **env_kwargs)
    obs, info = env.reset(seed=seed)  # Seed identique fixée

    trajectory = []
    frames = []
    total_reward = 0.0
    done = False

    while not done:
      action, _ = model.predict(obs, deterministic=True)

      # Capture et redimensionnement léger (304x200)
      frame = env.render()
      frame_resized = cv2.resize(frame, (304, 200))
      frames.append(frame_resized)

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
    del env

    video_b64 = ""
    if frames:
      with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "episode.mp4")
        imageio.mimsave(
            path,
            frames,
            fps=30,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        with open(path, "rb") as f:
          video_b64 = base64.b64encode(f.read()).decode("utf-8")

    key = "with_wind" if wind_enabled else "no_wind"
    results[key] = {
        "total_reward": total_reward,
        "steps": len(trajectory),
        "trajectory": trajectory,
        "video": video_b64,
    }

    del frames
    gc.collect()

  return {
      "seed": seed,
      "no_wind": results["no_wind"],
      "with_wind": results["with_wind"],
  }