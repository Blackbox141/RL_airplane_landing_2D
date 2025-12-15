# train_alt.py
import os

# MAC CRASH FIX
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from airplane_alt_env import AirplaneAltitudeEnv

# --- KONFIGURATION ---
BASE_DIR = "./airplane_project_data"
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VECNORM_DIR = os.path.join(BASE_DIR, "vecnormalize")


def get_next_run_id(save_path):
    if not os.path.exists(save_path):
        return 1
    max_id = 0
    for folder_name in os.listdir(save_path):
        if folder_name.startswith("run_"):
            try:
                current_id = int(folder_name.split("_")[1])
                if current_id > max_id:
                    max_id = current_id
            except:
                pass
    return max_id + 1


def make_env():
    return AirplaneAltitudeEnv(render_mode="none", enable_scenarios=True, max_episode_steps=3000)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VECNORM_DIR, exist_ok=True)

    run_id = get_next_run_id(MODEL_DIR)
    run_name = f"run_{run_id}"

    current_model_dir = os.path.join(MODEL_DIR, run_name)
    current_vecnorm_path = os.path.join(VECNORM_DIR, f"{run_name}_vecnorm.pkl")
    os.makedirs(current_model_dir, exist_ok=True)

    env = make_vec_env(make_env, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=os.path.join(BASE_DIR, "tensorboard"),
    )

    try:
        model.learn(total_timesteps=1_500_000, tb_log_name=run_name)
    except KeyboardInterrupt:
        print("Abbruch durch User...")

    final_path = os.path.join(current_model_dir, "model_final")
    model.save(final_path)
    env.save(current_vecnorm_path)

    print(f"Gespeichert: {final_path}.zip")
    print(f"VecNormalize: {current_vecnorm_path}")

    env.close()


if __name__ == "__main__":
    main()