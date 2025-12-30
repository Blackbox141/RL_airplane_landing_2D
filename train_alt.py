import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from airplane_alt_env import AirplaneAltitudeEnv

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

BASE_LOG_DIR = "altitude_agent"


def get_next_run_number(base_path):
    if not os.path.exists(base_path):
        return 1
    runs = [d for d in os.listdir(base_path) if d.startswith("run_")]
    if not runs:
        return 1
    numbers = []
    for r in runs:
        try:
            numbers.append(int(r.split("_")[1]))
        except Exception:
            continue
    return max(numbers) + 1 if numbers else 1


def make_env():
    return AirplaneAltitudeEnv(render_mode="none")


def train():
    run_id = get_next_run_number(BASE_LOG_DIR)
    run_folder = os.path.join(BASE_LOG_DIR, f"run_{run_id}")
    log_path = os.path.join(run_folder, "logs")
    os.makedirs(run_folder, exist_ok=True)

    print(f"🚀 Starte Training (Run {run_id}) auf M2")

    # Paralleles Training (8 Prozesse) – nutzt M2 besser aus
    env = make_vec_env(
        make_env,
        n_envs=8,
        vec_env_cls=SubprocVecEnv
    )

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=100.,
        training=False
    )

    class SyncVecNormalizeCallback(EvalCallback):
        def _on_step(self) -> bool:
            if hasattr(self.eval_env, 'ret_rms') and hasattr(self.training_env, 'ret_rms'):
                self.eval_env.ret_rms = self.training_env.ret_rms
            if hasattr(self.eval_env, 'obs_rms') and hasattr(self.training_env, 'obs_rms'):
                self.eval_env.obs_rms = self.training_env.obs_rms
            return super()._on_step()

    eval_callback = SyncVecNormalizeCallback(
        eval_env,
        best_model_save_path=run_folder,
        log_path=run_folder,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=1
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        normalize_advantage=True,
        tensorboard_log=log_path,
        verbose=1
    )

    total_timesteps = 10_000_000
    print(f"Training startet für {total_timesteps} Steps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_Altitude_Fixed"
    )

    model.save(os.path.join(run_folder, "model_final"))
    env.save(os.path.join(run_folder, "vecnorm.pkl"))

    print("✅ Training beendet.")
    return run_folder


if __name__ == "__main__":
    train()