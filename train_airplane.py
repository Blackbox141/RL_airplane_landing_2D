import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# Dein Environment importieren
from airplane_alt_env import AirplaneAltitudeEnv

# --- KONFIGURATION DER ORDNER ---
BASE_PATH = "altitude_agent"


def get_next_run_number():
    if not os.path.exists(BASE_PATH):
        return 1
    runs = [d for d in os.listdir(BASE_PATH) if d.startswith("run_")]
    if not runs:
        return 1
    numbers = [int(r.split("_")[1]) for r in runs if r.split("_")[1].isdigit()]
    return max(numbers) + 1 if numbers else 1


def make_env(render_mode="none"):
    return AirplaneAltitudeEnv(render_mode=render_mode)


def train():
    # 1. Dynamischen Pfad erstellen (z.B. altitude_agent/run_1)
    run_id = get_next_run_number()
    run_folder = os.path.join(BASE_PATH, f"run_{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    # Pfade für Speicherung
    best_model_path = os.path.join(run_folder, "best_model")  # Ordner für das beste Modell
    final_model_path = os.path.join(run_folder, "model_final")
    vecnorm_path = os.path.join(run_folder, "vecnorm.pkl")
    log_path = os.path.join(run_folder, "logs")

    print(f"🚀 Starte Training in: {run_folder}")

    # 2. Trainings-Umgebung (4 Envs parallel)
    env = make_vec_env(lambda: make_env(render_mode="none"), n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Evaluierungs-Umgebung (Wichtig für den Callback)
    # Hier testet der Callback den Agenten
    eval_env = make_vec_env(lambda: make_env(render_mode="none"), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 4. BEST MODEL CALLBACK definieren
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_folder,  # Speichert unter altitude_agent/run_X/best_model.zip
        log_path=run_folder,
        eval_freq=10000,  # Alle 10.000 Schritte testen (bei 4 Envs alle 2.500 Iterationen)
        deterministic=True,
        render=False
    )

    # 5. PPO Modell definieren
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path
    )

    # 6. Training starten (mit Callback)
    model.learn(total_timesteps=1_500_000, callback=eval_callback)

    # 7. Finalen Stand und Normalisierung speichern
    model.save(final_model_path)
    env.save(vecnorm_path)

    print(f"✅ Training beendet.")
    print(f"📍 Bestes Modell unter: {os.path.join(run_folder, 'best_model.zip')}")
    return run_folder, os.path.join(run_folder, "best_model.zip"), vecnorm_path


def evaluate(model_path, vecnorm_path):
    print(f"🔬 Teste das BESTE Modell: {model_path}")

    env = make_vec_env(lambda: AirplaneAltitudeEnv(render_mode="human"), n_envs=1)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path)

    obs = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        if dones[0]:
            obs = env.reset()

    env.close()


def main():
    # 1. Trainieren & Bestes Modell finden
    run_folder, best_m_path, v_path = train()

    # 2. Das beste Modell sofort evaluieren
    evaluate(best_m_path, v_path)


if __name__ == "__main__":
    main()