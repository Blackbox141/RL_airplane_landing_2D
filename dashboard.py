import streamlit as st
import pandas as pd
import os
import glob
import plotly.graph_objects as go
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- KONFIGURATION ---
# Wir greifen nur auf die Daten zu, wir importieren KEIN pygame oder Environment!
BASE_DIR = "altitude_agent"

st.set_page_config(page_title="Trainings Stats", layout="wide", page_icon="📊")

# CSS für ein sauberes Dark-Theme Design
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    stSubheader { color: #00FFCC; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 KI-Pilot Trainings-Statistiken")

# 1. Prüfen ob der Hauptordner existiert
if not os.path.exists(BASE_DIR):
    st.error(f"❌ Ordner '{BASE_DIR}' nicht gefunden.")
    st.info("Tipp: Starte erst das Training, damit der Ordner erstellt wird.")
    st.stop()

# 2. Alle verfügbaren Runs finden
run_folders = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))], reverse=True)

if not run_folders:
    st.warning("⚠️ Keine Runs im Ordner 'altitude_agent' gefunden.")
    st.stop()

# Auswahl in der Seitenleiste
selected_run = st.sidebar.selectbox("Wähle einen Run (Training):", run_folders)
st.sidebar.markdown("---")
st.sidebar.info("Diese Ansicht ist rein informativ und nutzt kein Pygame.")


# --- FUNKTION: TENSORBOARD DATEN LADEN ---
@st.cache_data(ttl=30)
def load_metrics(run_name):
    run_path = os.path.join(BASE_DIR, run_name)
    # Suche rekursiv nach der event-Datei (findet sie auch in Unterordnern wie PPO_1)
    event_files = glob.glob(os.path.join(run_path, "**", "events.out.tfevents*"), recursive=True)

    if not event_files:
        return None, f"Keine Statistik-Daten in {run_name} gefunden."

    latest_file = max(event_files, key=os.path.getctime)

    try:
        ea = EventAccumulator(latest_file)
        ea.Reload()
        tags = ea.Tags()['scalars']
        data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            data[tag] = pd.DataFrame({
                "Step": [e.step for e in events],
                "Value": [e.value for e in events]
            })
        return data, None
    except Exception as e:
        return None, str(e)


# --- VISUALISIERUNG ---
metrics, error = load_metrics(selected_run)

if error:
    st.error(f"Fehler beim Laden: {error}")
elif metrics:
    # 1. REWARD (Hauptfortschritt)
    if 'rollout/ep_rew_mean' in metrics:
        st.subheader("📈 Gesamter Lernfortschritt (Mean Reward)")
        df = metrics['rollout/ep_rew_mean']

        # Plotly Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Step'], y=df['Value'], mode='lines',
                                 name='Mean Reward', line=dict(color='#00FFCC', width=2)))

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Trainings-Schritte",
            yaxis_title="Belohnung (höher = besser)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2. DETAILS (Zweispaltig)
    col1, col2 = st.columns(2)

    with col1:
        # Entropy: Wie sicher ist sich die KI?
        if 'train/entropy_loss' in metrics:
            st.subheader("🧠 Entscheidungs-Sicherheit")
            df = metrics['train/entropy_loss']
            fig_ent = go.Figure(go.Scatter(x=df['Step'], y=df['Value'], line=dict(color='orange')))
            fig_ent.update_layout(template="plotly_dark", title="Entropy Loss")
            st.plotly_chart(fig_ent, use_container_width=True)
            st.caption("Sinkende Werte bedeuten, dass die KI eine feste Strategie entwickelt.")

    with col2:
        # Value Loss: Wie gut schätzt die KI die Situation ein?
        if 'train/value_loss' in metrics:
            st.subheader("📉 Vorhersage-Genauigkeit")
            df = metrics['train/value_loss']
            fig_val = go.Figure(go.Scatter(x=df['Step'], y=df['Value'], line=dict(color='red')))
            fig_val.update_layout(template="plotly_dark", title="Value Loss")
            st.plotly_chart(fig_val, use_container_width=True)
            st.caption("Niedrige Werte zeigen an, dass die KI Belohnungen korrekt vorhersagt.")

    # Rohdaten Liste
    with st.expander("Verfügbare Rohdaten-Tags anzeigen"):
        st.write(list(metrics.keys()))
else:
    st.info(
        "Warte auf Daten... Falls das Training gerade erst gestartet ist, dauert es ca. 1-2 Minuten bis die ersten Werte erscheinen.")