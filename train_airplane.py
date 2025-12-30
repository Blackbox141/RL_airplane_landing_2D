import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Dein Environment importieren
from airplane_alt_env import AirplaneAltitudeEnv

# MAC CRASH FIX
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# --- KONFIGURATION DER ORDNER ---
BASE_LOG_DIR = "altitude_agent"


def get_next_run_number(base_path):
    if not os.path.exists(base_path): return 1
    runs = [d for d in os.listdir(base_path) if d.startswith("run_")]
    if not runs: return 1
    numbers = []
    for r in runs:
        try:
            numbers.append(int(r.split("_")[1]))
        except:
            continue
    return max(numbers) + 1 if numbers else 1


def make_env():
    return AirplaneAltitudeEnv(render_mode="none")


def train():
    # 1. Setup Run-Ordner
    run_id = get_next_run_number(BASE_LOG_DIR)
    run_folder = os.path.join(BASE_LOG_DIR, f"run_{run_id}")
    log_path = os.path.join(run_folder, "logs")
    os.makedirs(run_folder, exist_ok=True)

    print(f"🚀 Starte Training mit Original-Parametern: {run_folder}")

    # 2. Haupt-Trainingsumgebung (4 Envs wie im Original)
    env = make_vec_env(make_env, n_envs=4)
    # Normalisierung wie in vecnorm.pkl [cite: 2, 3]
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Evaluierungs-Umgebung
    eval_env = make_vec_env(make_env, n_envs=1)
    # Statistiken werden synchronisiert, Training für Eval deaktiviert
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

    # 4. Custom Callback zur Synchronisation
    class SyncVecNormalizeCallback(EvalCallback):
        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            if hasattr(self.eval_env, 'ret_rms'):
                self.eval_env.ret_rms = self.training_env.ret_rms
            return super()._on_step()

    eval_callback = SyncVecNormalizeCallback(
        eval_env,
        best_model_save_path=run_folder,
        log_path=run_folder,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # 5. PPO Modell mit exakten Werten aus dem Best-Model [cite: 60, 64, 65, 74]
    model = PPO(
        "MlpPolicy",
        env,

        tensorboard_log=log_path
    )

    # Training starten (Original lief ca. 1.5 Mio Schritte) [cite: 75]
    model.learn(total_timesteps=1_500_000, callback=eval_callback, tb_log_name="PPO")

    # 6. Finale Speicherung
    model.save(os.path.join(run_folder, "model_final"))
    env.save(os.path.join(run_folder, "vecnorm.pkl"))

    print(f"✅ Training beendet. Bestes Modell liegt in: {run_folder}/best_model.zip")
    return run_folder


def main():
    run_folder = train()
    print(f"Modell und Normalisierung gespeichert in {run_folder}")


if __name__ == "__main__":
    main()