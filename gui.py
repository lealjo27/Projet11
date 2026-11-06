import base64
import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="LunarLander-v3 Agent - GUI Simulation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 LunarLander-v3 Agent - Interface de Simulation")
st.markdown(
    "Interface graphique interactive interrogeant l'API FastAPI backend "
    "pour simuler, contrôler le vent et analyser le comportement du modèle DQN."
)

# ------------------------------------------------------------------------------
# BARRE LATÉRALE DE COMMANDES
# ------------------------------------------------------------------------------
st.sidebar.header("🕹️ Configuration de la Simulation")

# Réglage du vent
# enable_wind = st.sidebar.checkbox("Activer le vent 💨", value=False)
# wind_power = 15.0
# if enable_wind:
#     wind_power = st.sidebar.slider(
#         "Force du vent (wind_power)",
#         min_value=1.0,
#         max_value=30.0,
#         value=15.0,
#         step=1.0,
#     )

st.sidebar.divider()

# # Bouton 1 : Partie Simple
# if st.sidebar.button("🎮 Lancer une nouvelle partie (/play)", type="primary"):
#     with st.spinner("Demande de trajectoire à l'API..."):
#         try:
#             # params = {"enable_wind": enable_wind, "wind_power": wind_power}
#             response = requests.get(f"{API_URL}/play",  timeout=60)
#             if response.status_code == 200:
#                 st.session_state["play_data"] = response.json()
#                 st.session_state.pop("compare_data", None)
#                 st.session_state["show_stats"] = False
#                 st.success("Partie récupérée avec succès !")
#             else:
#                 st.error(f"Erreur API ({response.status_code}): {response.text}")
#         except Exception as e:
#             st.error(f"Impossible de contacter l'API sur {API_URL}: {e}")


st.sidebar.header("⚖️ Mode Comparaison")
wind_power_comp = st.sidebar.select_slider(
        "Force du vent (wind_power)",
        options=[0, 5, 15, 20, 25, 30],
        value=15
    )

lock_seed = st.sidebar.checkbox("🔒 Garder la même partie (Même Seed)", value=True)
# Bouton 2 : Comparaison
if st.sidebar.button("⚔️ Comparer Même Partie (Sans Vent vs Avec Vent)"):
    with st.spinner("Exécution des 2 simulations (graine/seed identique)..."):
        try:
            # Préparation des paramètres HTTP
            params = {"wind_power": wind_power_comp}
            
            # Si l'option est cochée ET qu'une seed existe déjà en mémoire, on la transmet !
            if lock_seed and "current_seed" in st.session_state:
                params["seed"] = st.session_state["current_seed"]


            # Appel API
            res = requests.get(f"{API_URL}/play_compare", params=params, timeout=90)
            
            if res.status_code == 200:
                            data = res.json()
                            st.session_state["compare_data"] = data
                            # On enregistre la seed reçue de l'API pour les clics futurs
                            st.session_state["current_seed"] = data["seed"]
                            st.success(f"Parties comparées avec succès (Seed #{data['seed']}) !")
            else:
                            st.error(f"Erreur API : {res.text}")
        except Exception as e:
                        st.error(f"Impossible d'atteindre l'API : {e}")

# ------------------------------------------------------------------------------
# FONCTION HELPER : DASHBOARD ANALYTIQUE PLOTLY (Reutilisable)
# ------------------------------------------------------------------------------
def generer_dashboard_analytique(data_partie, suffixe_titre=""):
    trajectory = data_partie["trajectory"]
    df = pd.DataFrame(trajectory)
    df["Pas"] = df.index + 1
    df["cumulative_reward"] = df["reward"].cumsum()

    actions_map = {
        0: "Ne rien faire (Inaction)",
        1: "Moteur Gauche",
        2: "Moteur Principal",
        3: "Moteur Droit",
    }
    df["Action_Label"] = df["action"].map(actions_map)

    df["pos_x"] = df["state"].apply(lambda s: s[0])
    df["pos_y"] = df["state"].apply(lambda s: s[1])

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

    # 1. Camembert des Actions
    st.subheader(f"📊 Prise de Décision de l'IA {suffixe_titre}")
    fig_pie = px.pie(
        df,
        names="Action_Label",
        title=f"Distribution des actions ({suffixe_titre})",
        hole=0.3,
    )
    st.plotly_chart(fig_pie, width='stretch')

    # 2. Progression du score
    st.subheader(f"📈 Évolution du Score Cumulé {suffixe_titre}")
    fig_line = px.line(
        df,
        x="Pas",
        y="cumulative_reward",
        title=f"Progression du score ({suffixe_titre})",
        labels={"cumulative_reward": "Score Cumulé", "Pas": "Pas de temps"},
        hover_data=["reward", "Action_Label"],
    )
    fig_line.update_traces(line_color="#1f77b4", line_width=3)
    st.plotly_chart(fig_line, width='stretch')

    # 3. Trajectoire 2D
    st.subheader(f"🗺️ Trajectoire 2D du Module Spatial {suffixe_titre}")
    fig_scatter = px.scatter(
        df,
        x="pos_x",
        y="pos_y",
        color="Action_Label",
        title=f"Profil de Vol ({suffixe_titre})",
        labels={
            "pos_x": "Position X",
            "pos_y": "Altitude Y",
            "Action_Label": "Action AI",
        },
        hover_data=["Pas", "reward"],
        range_x=[-1, 1],
        range_y=[-0.1, 1.5],
    )
    fig_scatter.add_shape(
        type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="Gray", width=2)
    )
    fig_scatter.add_shape(
        type="rect",
        x0=-0.2,
        y0=-0.02,
        x1=0.2,
        y1=0,
        fillcolor="Green",
        opacity=0.6,
        line_width=0,
    )
    st.plotly_chart(fig_scatter, width='stretch')

    st.divider()

    # 4. Histogrammes par Circonstance
    st.subheader(f"📊 Décisions par Circonstance de Vol {suffixe_titre}")
    col_adv1, col_adv2 = st.columns(2)

    with col_adv1:
        st.markdown("**Selon l'Altitude**")
        fig_bar_alt = px.histogram(
            df,
            x="Altitude_Cat",
            color="Action_Label",
            title="Distribution par Altitude",
            labels={
                "Altitude_Cat": "Circonstance (Altitude)",
                "count": "Nombre de Pas",
            },
            barmode="stack",
        )
        st.plotly_chart(fig_bar_alt, width='stretch')

    with col_adv2:
        st.markdown("**Selon la Vitesse Verticale**")
        fig_bar_vit = px.histogram(
            df,
            x="Vitesse_Cat",
            color="Action_Label",
            title="Distribution par Vitesse Verticale",
            labels={
                "Vitesse_Cat": "Circonstance (Vitesse Vy)",
                "count": "Nombre de Pas",
            },
            barmode="group",
        )
        st.plotly_chart(fig_bar_vit, width='stretch')

    st.divider()

    # 5. Moyenne et Écart-type des Récompenses
    st.subheader(
        f"📈 Moyenne & Écart-type des Récompenses par Altitude {suffixe_titre}"
    )
    stats_df = (
        df.groupby("Altitude_Cat")["reward"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats_df.columns = [
        "Circonstance (Altitude)",
        "Récompense Moyenne",
        "Écart-Type (Std)",
        "Nombre de Pas",
    ]
    stats_df["Écart-Type (Std)"] = stats_df["Écart-Type (Std)"].fillna(0)

    fig_stats = px.bar(
        stats_df,
        x="Circonstance (Altitude)",
        y="Récompense Moyenne",
        error_y="Écart-Type (Std)",
        title="Récompense Moyenne et Variance par Altitude",
        text_auto=".2f",
        color="Circonstance (Altitude)",
    )
    st.plotly_chart(fig_stats, width='stretch')
    st.dataframe(stats_df, width='stretch')


# ------------------------------------------------------------------------------
# AFFICHAGE MODE SIMPLE (/play)
# ------------------------------------------------------------------------------
if "play_data" in st.session_state:
    data = st.session_state["play_data"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Score total", f"{data['total_reward']:.2f}")
    col2.metric("Nombre de pas", data["steps"])
    col3.metric(
        "Statut final",
        "Atterrissage réussi" if data["total_reward"] > 200 else "Crash / Échec",
    )

    st.divider()
    st.subheader("📹 Animation Vidéo de la partie")

    if "video" in data and data["video"]:
        video_b64 = data["video"]
        video_html = f"""
            <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <video style="width: 100%; max-height: 400px; object-fit: contain;" autoplay loop muted playsinline controls>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            Votre navigateur ne supporte pas la lecture de cette vidéo.
        </video>
    </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)

    st.divider()
    if st.button("📊 Voir les stats", type="secondary"):
        st.session_state["show_stats"] = True

    if st.session_state.get("show_stats", False):
        generer_dashboard_analytique(data)
# ==============================================================================
# INTEGRATION STREAMLIT : Mode Comparaison Entrelacé
# ==============================================================================
if "compare_data" in st.session_state:
    cmp = st.session_state["compare_data"]
    st.subheader(f"🔍 Comparaison de la Partie (Seed #{cmp['seed']})")

    # 1. Alignement des Métriques et Vidéos côte à côte
    col_nw, col_w = st.columns(2)

    with col_nw:
        st.markdown("### 🛑 Sans Vent")
        data_nw = cmp["no_wind"]
        st.metric(
            "Score Total",
            f"{data_nw['total_reward']:.2f}",
            delta="Référence",
            delta_color="off",
        )
        st.metric(
            "Nombre de Pas",
            data_nw["steps"],
            delta="Référence",
            delta_color="off",
        )

        if data_nw.get("video"):
            v_b64 = data_nw["video"]
            v_html = f"""
                <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                    <video style="width: 100%; max-height: 400px; object-fit: contain;"   muted playsinline controls>
                        <source src="data:video/mp4;base64,{v_b64}" type="video/mp4">
                        Votre navigateur ne supporte pas la lecture de cette vidéo.
                    </video>
                </div>
            """
            st.markdown(v_html, unsafe_allow_html=True)

    with col_w:
        st.markdown("### 💨 Avec Vent")
        data_w = cmp["with_wind"]
        diff_reward = data_w["total_reward"] - data_nw["total_reward"]
        diff_steps = data_w["steps"] - data_nw["steps"]

        st.metric(
            "Score Total",
            f"{data_w['total_reward']:.2f}",
            delta=f"{diff_reward:.2f}",
        )
        st.metric("Nombre de Pas", data_w["steps"], delta=f"{diff_steps}")

        if data_w.get("video"):
            v_b64 = data_w["video"]
            v_html = f"""
                <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                    <video style="width: 100%; max-height: 400px; object-fit: contain;"   muted playsinline controls>
                        <source src="data:video/mp4;base64,{v_b64}" type="video/mp4">
                        Votre navigateur ne supporte pas la lecture de cette vidéo.
                    </video>
                </div>
            """
            st.markdown(v_html, unsafe_allow_html=True)

    st.divider()

    # 2. Bouton Déclencheur du Dashboard Comparatif
    if st.button("📊 Voir les stats comparatives", type="secondary"):
        st.session_state["show_compare_stats"] = True

    # 3. Affichage Entrelacé des Graphiques (Côte à côte par type de graphique)
    if st.session_state.get("show_compare_stats", False):
        # ----------------------------------------------------------------------
        # PRÉPARATION DES DATAFRAMES
        # ----------------------------------------------------------------------
        def preparer_df(data_partie):
            df = pd.DataFrame(data_partie["trajectory"])
            df["Pas"] = df.index + 1
            df["cumulative_reward"] = df["reward"].cumsum()
            actions_map = {
                0: "Ne rien faire (Inaction)",
                1: "Moteur Gauche",
                2: "Moteur Principal",
                3: "Moteur Droit",
            }
            df["Action_Label"] = df["action"].map(actions_map)
            df["pos_x"] = df["state"].apply(lambda s: s[0])
            df["pos_y"] = df["state"].apply(lambda s: s[1])

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
            return df

        df_nw = preparer_df(cmp["no_wind"])
        df_w = preparer_df(cmp["with_wind"])

        # ----------------------------------------------------------------------
        # 1. DISTRIBUTION DES ACTIONS (PIE CHART)
        # ----------------------------------------------------------------------
        st.markdown("## 📊 1. Distribution des Actions (Prise de Décision)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🛑 Sans Vent")
            fig_pie_nw = px.pie(
                df_nw,
                names="Action_Label",
                color="Action_Label",
                title="Distribution des actions (Sans Vent)",
                hole=0.3,
            )
            st.plotly_chart(fig_pie_nw, width='stretch')
        with c2:
            st.markdown("#### 💨 Avec Vent")
            fig_pie_w = px.pie(
                df_w,
                names="Action_Label",
                color="Action_Label",
                title="Distribution des actions (Avec Vent)",
                hole=0.3,
            )
            st.plotly_chart(fig_pie_w, width='stretch')

        st.divider()

        # ----------------------------------------------------------------------
        # 2. ÉVOLUTION DU SCORE CUMULÉ (LINE CHART)
        # ----------------------------------------------------------------------
        st.markdown("## 📈 2. Évolution du Score Cumulé")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🛑 Sans Vent")
            fig_line_nw = px.line(
                df_nw,
                x="Pas",
                y="cumulative_reward",
                title="Score cumulé (Sans Vent)",
                labels={"cumulative_reward": "Score Cumulé", "Pas": "Pas de temps"},
                hover_data=["reward", "Action_Label"],
            )
            fig_line_nw.update_traces(line_color="#1f77b4", line_width=3)
            st.plotly_chart(fig_line_nw, width='stretch')
        with c2:
            st.markdown("#### 💨 Avec Vent")
            fig_line_w = px.line(
                df_w,
                x="Pas",
                y="cumulative_reward",
                title="Score cumulé (Avec Vent)",
                labels={"cumulative_reward": "Score Cumulé", "Pas": "Pas de temps"},
                hover_data=["reward", "Action_Label"],
            )
            fig_line_w.update_traces(line_color="#ef553b", line_width=3)
            st.plotly_chart(fig_line_w, width='stretch')

        st.divider()

        # ----------------------------------------------------------------------
        # 3. TRAJECTOIRE 2D (SCATTER PLOT)
        # ----------------------------------------------------------------------
        st.markdown("## 🗺️ 3. Trajectoire 2D et Profil de Vol")
        c1, c2 = st.columns(2)
        
        def tracer_scatter(df_target, titre):
            fig = px.scatter(
                df_target,
                x="pos_x",
                y="pos_y",
                color="Action_Label",
                title=titre,
                labels={"pos_x": "Position X", "pos_y": "Altitude Y", "Action_Label": "Action AI"},
                hover_data=["Pas", "reward"],
                range_x=[-1, 1],
                range_y=[-0.1, 1.5],
            )
            fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="Gray", width=2))
            fig.add_shape(type="rect", x0=-0.2, y0=-0.02, x1=0.2, y1=0, fillcolor="Green", opacity=0.6, line_width=0)
            return fig

        with c1:
            st.markdown("#### 🛑 Sans Vent")
            st.plotly_chart(tracer_scatter(df_nw, "Profil de Vol (Sans Vent)"), width='stretch')
        with c2:
            st.markdown("#### 💨 Avec Vent")
            st.plotly_chart(tracer_scatter(df_w, "Profil de Vol (Avec Vent)"), width='stretch')

        st.divider()

        # ----------------------------------------------------------------------
        # 4. DÉCISIONS PAR ALTITUDE (HISTOGRAM)
        # ----------------------------------------------------------------------
        st.markdown("## 📊 4. Décisions par Tranche d'Altitude")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🛑 Sans Vent")
            fig_bar_nw = px.histogram(
                df_nw,
                x="Altitude_Cat",
                color="Action_Label",
                title="Actions par Altitude (Sans Vent)",
                barmode="stack",
            )
            st.plotly_chart(fig_bar_nw, width='stretch')
        with c2:
            st.markdown("#### 💨 Avec Vent")
            fig_bar_w = px.histogram(
                df_w,
                x="Altitude_Cat",
                color="Action_Label",
                title="Actions par Altitude (Avec Vent)",
                barmode="stack",
            )
            st.plotly_chart(fig_bar_w, width='stretch')

        st.divider()

        # ----------------------------------------------------------------------
        # 5. MOYENNE ET ÉCART-TYPE DES RÉCOMPENSES (BAR CHART)
        # ----------------------------------------------------------------------
        st.markdown("## 📈 5. Moyenne & Écart-type des Récompenses par Altitude")
        c1, c2 = st.columns(2)

        def calculer_stats(df_target):
            stats = df_target.groupby("Altitude_Cat")["reward"].agg(["mean", "std", "count"]).reset_index()
            stats.columns = ["Circonstance (Altitude)", "Récompense Moyenne", "Écart-Type (Std)", "Nombre de Pas"]
            stats["Écart-Type (Std)"] = stats["Écart-Type (Std)"].fillna(0)
            return stats

        stats_nw = calculer_stats(df_nw)
        stats_w = calculer_stats(df_w)

        with c1:
            st.markdown("#### 🛑 Sans Vent")
            fig_stats_nw = px.bar(
                stats_nw,
                x="Circonstance (Altitude)",
                y="Récompense Moyenne",
                error_y="Écart-Type (Std)",
                title="Moyenne & Variance (Sans Vent)",
                text_auto=".2f",
                color="Circonstance (Altitude)",
            )
            st.plotly_chart(fig_stats_nw, width='stretch')
            st.dataframe(stats_nw, width='stretch')

        with c2:
            st.markdown("#### 💨 Avec Vent")
            fig_stats_w = px.bar(
                stats_w,
                x="Circonstance (Altitude)",
                y="Récompense Moyenne",
                error_y="Écart-Type (Std)",
                title="Moyenne & Variance (Avec Vent)",
                text_auto=".2f",
                color="Circonstance (Altitude)",
            )
            st.plotly_chart(fig_stats_w, width='stretch')
            st.dataframe(stats_w, width='stretch')