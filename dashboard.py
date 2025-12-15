import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- KONFIGURATION ---
BASE_DIR = "./airplane_project_data"
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "tensorboard")

st.set_page_config(page_title="RL Flight Analyzer", layout="wide", page_icon="✈️")
st.title("✈️ Airplane Autopilot Analyzer")

# --- 1. DATEN STRUKTUR LADEN ---
if not os.path.exists(TENSORBOARD_DIR):
    st.error(f"Ordner {TENSORBOARD_DIR} nicht gefunden. Bitte trainiere erst.")
    st.stop()

# Runs aus TensorBoard ableiten (robust, auch wenn CSV fehlt)
tb_runs = set()
for name in os.listdir(TENSORBOARD_DIR):
    if name.startswith("run_"):
        base = name.split("_")[0] + "_" + name.split("_")[1]
        tb_runs.add(base)

run_folders = sorted(
    tb_runs,
    key=lambda x: int(x.split("_")[1]),
    reverse=True
)

if not run_folders:
    st.warning("Keine Runs gefunden.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("🕹️ Steuerung")
selected_run = st.sidebar.selectbox("1. Wähle Run:", run_folders)  # z.B. "run_3"

# Pfade
current_log_dir = os.path.join(LOG_DIR, selected_run)
current_model_dir = os.path.join(MODEL_DIR, selected_run)

# Modelle für Selektion laden
model_files = glob.glob(os.path.join(current_model_dir, "*.zip"))


def get_steps(filename):
    name = os.path.basename(filename)
    if "final" in name: return 999999999
    try:
        parts = name.split("_")
        idx = parts.index("steps")
        return int(parts[idx - 1])
    except:
        return 0


model_files.sort(key=get_steps, reverse=True)
model_names = [os.path.basename(f) for f in model_files]
selected_model_name = st.sidebar.selectbox("2. Wähle Checkpoint:", model_names) if model_names else None


# --- FUNKTION: TENSORBOARD DATEN LADEN ---
@st.cache_data(ttl=60)  # Cacht die Daten für 60 Sekunden für Performance
def load_tensorboard_data(run_name):
    # Stable Baselines nennt den TB Ordner oft "run_3_1" statt nur "run_3"
    # Wir suchen nach Ordnern, die mit run_name beginnen
    search_path = os.path.join(TENSORBOARD_DIR, run_name + "*")
    found_folders = glob.glob(search_path)

    if not found_folders:
        return None, "Ordner nicht gefunden"

    # Nimm den neuesten Ordner (falls mehrere existieren, z.B. run_3_1, run_3_2)
    tb_folder = sorted(found_folders)[-1]

    # Suche das event file
    event_files = glob.glob(os.path.join(tb_folder, "events.out.tfevents*"))
    if not event_files:
        return None, "Kein Event File gefunden"

    # Nimm das größte/neueste File
    event_file = max(event_files, key=os.path.getctime)

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        # Verfügbare Tags extrahieren
        tags = ea.Tags()['scalars']

        data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = pd.DataFrame({"step": steps, "value": values})

        return data, None
    except Exception as e:
        return None, str(e)


# --- DATEN LADEN (CSV & EVAL) ---
# ... (Code für CSV laden wie vorher, hier verkürzt dargestellt) ...
monitor_files = glob.glob(os.path.join(current_log_dir, "*.monitor.csv"))
df_train = pd.DataFrame()
if monitor_files:
    dfs = []
    for f in monitor_files:
        try:
            if os.stat(f).st_size > 100:
                df = pd.read_csv(f, skiprows=1)
                dfs.append(df)
        except:
            pass
    if dfs:
        df_train = pd.concat(dfs, ignore_index=True)
        if 't' in df_train: df_train = df_train.sort_values('t')
        df_train['Reward_MA'] = df_train['r'].rolling(window=50).mean()

# --- HAUPTANSICHT ---

tab1, tab2, tab3 = st.tabs(["📈 Lernkurve (CSV)", "🔬 Checkpoint Analyse", "🧠 TensorBoard (Deep Dive)"])

with tab1:
    st.subheader(f"Basis-Metriken: {selected_run}")
    if df_train.empty:
        st.warning("⚠️ Keine CSV-Logs gefunden (dies passiert z.B. bei einem Absturz). Schau in Tab 3!")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train.index, y=df_train['Reward_MA'], mode='lines', name='Reward (Smoothed)'))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if selected_model_name:
        st.subheader(f"Checkpoint: {selected_model_name}")
        abs_path = os.path.abspath(os.path.join(current_model_dir, selected_model_name))
        st.code(f'run_simulation(r"{abs_path}")', language="python")
        st.info("Kopiere diesen Befehl in flight_simulation.py um genau diesen Stand zu testen.")
    else:
        st.info("Keine Modelle gefunden.")

with tab3:
    st.subheader("🧠 TensorBoard Daten (Das Gehirn des Agenten)")

    tb_data, error = load_tensorboard_data(selected_run)

    if error:
        st.error(f"Konnte TensorBoard Daten nicht laden: {error}")
        st.info("Hast du tensorboard installiert? (pip install tensorboard)")
    elif not tb_data:
        st.warning("Keine Daten im Event-File gefunden.")
    else:
        # 1. REWARD (Das wichtigste)
        st.markdown("### 1. Reward (aus TensorBoard)")
        if 'rollout/ep_rew_mean' in tb_data:
            df_rew = tb_data['rollout/ep_rew_mean']
            fig_tb = go.Figure()
            fig_tb.add_trace(go.Scatter(x=df_rew['step'], y=df_rew['value'], mode='lines', line=dict(color='#FF4B4B')))
            fig_tb.update_layout(xaxis_title="Timesteps", yaxis_title="Mean Reward", title="Episodic Reward Mean")
            st.plotly_chart(fig_tb, use_container_width=True)
        else:
            st.info("Keine Reward-Daten verfügbar.")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 2. Wie 'sicher' ist der Agent? (Entropy)")
            # Entropy Loss
            if 'train/entropy_loss' in tb_data:
                df_ent = tb_data['train/entropy_loss']
                fig_ent = go.Figure()
                fig_ent.add_trace(
                    go.Scatter(x=df_ent['step'], y=df_ent['value'], mode='lines', line=dict(color='orange')))
                fig_ent.update_layout(title="Entropy Loss (Neugierde)", xaxis_title="Timesteps")
                st.plotly_chart(fig_ent, use_container_width=True)
                st.caption(
                    "Fällt die Kurve? Gut! (Er lernt eine Strategie). Fällt sie zu schnell? Schlecht (Er probiert nix mehr).")

        with col_right:
            st.markdown("### 3. Versteht er die Situation? (Value Loss)")
            if 'train/value_loss' in tb_data:
                df_val = tb_data['train/value_loss']
                fig_val = go.Figure()
                fig_val.add_trace(
                    go.Scatter(x=df_val['step'], y=df_val['value'], mode='lines', line=dict(color='green')))
                fig_val.update_layout(title="Value Loss (Vorhersagefehler)", xaxis_title="Timesteps")
                st.plotly_chart(fig_val, use_container_width=True)
                st.caption("Ein niedriger Wert bedeutet: Er kann gut vorhersagen, wie viel Reward er gleich bekommt.")

        with st.expander("Alle verfügbaren Metriken anzeigen"):
            st.write("Gefundene Tags:", list(tb_data.keys()))