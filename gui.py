import os
import time
import requests
import streamlit as st
import plotly.express as px
import base64
import pandas as pd



API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LunarLander-v3 Agent - GUI simulation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("LunarLander-v3 Agent - GUI simulation")
st.markdown(
    "Cette interface graphique interroge l'API backend (`FastAPI`) "
    "pour récupérer et rejouer les trajectoires générées par l'agent DQN."
)

# Barre latérale de commande
st.sidebar.header("Commandes")

if st.sidebar.button("🎮 Lancer une nouvelle partie (/play)", type="primary"):
    with st.spinner("Demande d'une trajectoire à l'API..."):
        try:
            # Requête GET envoyée à l'endpoint /play de FastAPI
            response = requests.get(f"{API_URL}/play", timeout=10)
            
            if response.status_code == 200:
                st.session_state["play_data"] = response.json()
                st.success("Partie récupérée avec succès depuis l'API !")
            else:
                st.error(f"Erreur API ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Impossible de contacter l'API sur {API_URL}. Erreur : {e}")



# Affichage de la simulation si une partie a été chargée dans session_state
if "play_data" in st.session_state:
    data = st.session_state["play_data"]

    trajectory = data["trajectory"]
    df = pd.DataFrame(trajectory)
    df["Pas"] = df.index + 1
    df["cumulative_reward"] = df["reward"].cumsum() 
    # print(df.info())
    
    # 1. Métriques globales en haut de page
    col1, col2, col3 = st.columns(3)
    col1.metric("Score total", f"{data['total_reward']:.2f}")
    col2.metric("Nombre de pas (steps)", data['steps'])
    col3.metric("Statut final", "Atterrissage réussi" if data['total_reward'] > 200 else "Crash / Échec")
    
    st.divider()
    
    # 2. Animation pas à pas de la partie
    st.subheader("Animation de la partie")
    
    speed = st.slider("Vitesse de lecture (secondes par pas)", 0.01, 0.2, 0.05, 0.01)
    

    if st.button("▶️ Jouer l'animation vidéo"):
        image_spot = st.empty()
        status_spot = st.empty()
        
        actions_map = {0: "Inaction", 1: "Moteur Gauche", 2: "Moteur Principal", 3: "Moteur Droit"}
        
        for idx, step in enumerate(data["trajectory"]):
            # Si la frame image est disponible dans le JSON
            if "frame" in step:
                img_bytes = base64.b64decode(step["frame"])
                image_spot.image(img_bytes, width='stretch')
                
            status_spot.info(
                f"Pas {idx+1}/{data['steps']} | "
                f"Action : **{actions_map.get(step['action'], 'Inconnue')}** | "
                f"Récompense : {step['reward']:.2f}"
            )
            time.sleep(speed)

    if st.button("▶ Voir les stats", type="secondary"): 
            st.session_state["show_stats"] = True
    if st.session_state.get("show_stats", False):
# 1. Camembert de répartition des actions
        st.subheader("📊 Prise de Décision de l'IA")
        actions_map = {0: "Ne rien faire (Inaction)", 1: "Moteur Gauche", 2: "Moteur Principal", 3: "Moteur Droit"}
        df["Action_Label"] = df["action"].map(actions_map)

        fig_pie = px.pie(
            df, 
            names="Action_Label", 
            title="Distribution des actions pendant la partie",
            hole=0.3
        )
        st.plotly_chart(fig_pie, width='stretch')

        # 2. Courbe de progression de la récompense cumulée
        st.subheader("📈 Évolution du Score Cumulé")
        
        fig_line = px.line(
            df, 
            x="Pas", 
            y="cumulative_reward",
            title="Progression de la récompense cumulée au fil des pas",
            labels={"cumulative_reward": "Score Cumulé", "Pas": "Pas de temps"},
            hover_data=["reward", "Action_Label"]
        )
        fig_line.update_traces(line_color="#1f77b4", line_width=3)
        
        st.plotly_chart(fig_line, width='stretch')

        # Extraction des coordonnées X et Y pour le graphique 2D
        df["pos_x"] = df["state"].apply(lambda s: s[0])
        df["pos_y"] = df["state"].apply(lambda s: s[1])

        # 3. Nuage de points de la trajectoire 2D (Position X vs Altitude Y)
        st.subheader("🗺️ Trajectoire 2D du Module Spatial")
        
        fig_scatter = px.scatter(
            df, 
            x="pos_x", 
            y="pos_y", 
            color="Action_Label",
            title="Profil de Vol (Position Horizontale X vs Altitude Y)",
            labels={"pos_x": "Position X", "pos_y": "Altitude Y", "Action_Label": "Action AI"},
            hover_data=["Pas", "reward"],
            range_x=[-1, 1],
            range_y=[-0.1, 1.5]
        )
        
        # Repères visuels du sol et de la zone d'atterrissage
        fig_scatter.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="Gray", width=2))
        fig_scatter.add_shape(type="rect", x0=-0.2, y0=-0.02, x1=0.2, y1=0, fillcolor="Green", opacity=0.6, line_width=0)
        
        st.plotly_chart(fig_scatter, width='stretch')

        st.divider()
        st.subheader("📊 Analyse Avancée : Décisions de l'IA par Circonstance")

        # 1. Catégorisation des circonstances (Altitude, Vitesse verticale, Inclinaison)
        def categoriser_altitude(y):
            if y > 0.8:
                return "1. Haute Altitude"
            elif y > 0.3:
                return "2. Moyenne Altitude"
            else:
                return "3. Phase d'Atterrissage"

        def categoriser_vitesse(vy):
            if vy < -0.3:
                return "Chute Rapide"
            elif vy > 0.1:
                return "Remontée / Stabilisation"
            else:
                return "Vitesse Modérée"

        df["Altitude_Cat"] = df["pos_y"].apply(categoriser_altitude)
        df["Vitesse_Cat"] = df["state"].apply(lambda s: categoriser_vitesse(s[3]))

        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            st.markdown("**1. Décisions de l'IA selon l'Altitude (Barres Empilées)**")
            fig_bar_alt = px.histogram(
                df, 
                x="Altitude_Cat", 
                color="Action_Label", 
                title="Distribution des Actions par Tranche d'Altitude",
                labels={"Altitude_Cat": "Circonstance (Altitude)", "count": "Nombre de Pas"},
                barmode="stack"
            )
            st.plotly_chart(fig_bar_alt, width="stretch")

        with col_adv2:
            st.markdown("**2. Décisions selon la Vitesse Verticale**")
            fig_bar_vit = px.histogram(
                df, 
                x="Vitesse_Cat", 
                color="Action_Label", 
                title="Distribution des Actions selon la Vitesse Verticale",
                labels={"Vitesse_Cat": "Circonstance (Vitesse Vy)", "count": "Nombre de Pas"},
                barmode="group"
            )
            st.plotly_chart(fig_bar_vit, width="stretch")

        st.divider()

        # 2. Statistiques Détaillées : Moyenne et Écart-Type par Type de Circonstance
        st.subheader("📈 Moyenne et Écart-type des Récompenses par Circonstance")
        
        stats_df = df.groupby("Altitude_Cat")["reward"].agg(["mean", "std", "count"]).reset_index()
        stats_df.columns = ["Circonstance (Altitude)", "Récompense Moyenne", "Écart-Type (Std)", "Nombre de Pas"]
        stats_df["Écart-Type (Std)"] = stats_df["Écart-Type (Std)"].fillna(0)

        # Graphique d'erreur Plotly (Moyenne + Écart-Type)
        fig_stats = px.bar(
            stats_df, 
            x="Circonstance (Altitude)", 
            y="Récompense Moyenne", 
            error_y="Écart-Type (Std)",
            title="Récompense Moyenne et Écart-Type (Variance) par Altitude",
            text_auto=".2f",
            color="Circonstance (Altitude)"
        )
        st.plotly_chart(fig_stats, width="stretch")

        # Tableau synthétique des chiffres
        st.dataframe(stats_df, width="stretch")