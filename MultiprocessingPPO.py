import datetime as dt
import os

import gymnasium as gym
import minigrid
import numpy as np

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from InstantiateEnvs import make_env

def train_multiprocessing_ppo(main_config=None):
    """
    Train a PPO agent with multiprocessing.
    """
    assert main_config is not None, "main_config cannot be None"
    assert "env" in main_config, "main_config must contain 'env' key"
    assert "id" in main_config["env"], "main_config must contain 'id' key within 'env'"
    assert "wrappers" in main_config, "main_config must contain 'wrappers' key"
    assert "learn" in main_config, "main_config must contain 'learn' key"
    assert "checkpoints" in main_config, "main_config must contain 'checkpoints' key"

    env_id = main_config["env"]["id"]

    monitor_runs = main_config["wrappers"].get("monitor", False)
    flat_obs = main_config["wrappers"].get("flat_obs", False)
    img_obs = main_config["wrappers"].get("img_obs", False)
    num_cpu = main_config["env"].get("num_cpu", 1)

    training_id = f"PPO_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    result_filepath = main_config["env"].get("monitor_filepath", None)

    if result_filepath is not None:
        result_filepath = os.path.join(result_filepath, training_id)
        os.makedirs(result_filepath, exist_ok=True)
        monitor_runs = True
    else:
        monitor_runs = False
    
    result_filename = main_config["env"].get("monitor_filename", "monitor")

    n_timesteps = main_config["learn"].get("n_timesteps", 50_000)
    eval_episodes = main_config["learn"].get("eval_episodes", 10)

    checkpoint_load_path = main_config["checkpoints"].get("load", None)

    checkpoint_save_path = main_config["checkpoints"].get("save", None)
    checkpoint_save_path = os.path.join(checkpoint_save_path, training_id) if checkpoint_save_path is not None else None

    if checkpoint_save_path is not None and not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)

    save_freq = main_config["checkpoints"].get("save_freq", 10000)

    tensorboard_log = None
    if monitor_runs:
        tensorboard_log = os.path.join("tensorboard", env_id)
        if not os.path.exists(tensorboard_log):
            os.makedirs(tensorboard_log)

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i, path=result_filepath, filename=result_filename, monitor=monitor_runs, flat_obs=flat_obs, img_obs=img_obs) for i in range(num_cpu)])

    # print(env.observation_space)
    # exit()

    model = PPO(env=env, tensorboard_log=tensorboard_log, **main_config.get("model", {}))

    eval_env = make_env(env_id, 0, monitor=False, flat_obs=flat_obs, img_obs=img_obs)()

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


    if save_freq < n_timesteps:
        learn_loop = n_timesteps // save_freq
        n_timesteps = save_freq
    else:
        learn_loop = 1

    if checkpoint_load_path is not None:
        if os.path.exists(checkpoint_load_path):
            model = PPO.load(checkpoint_load_path, env=env, tensorboard_log=tensorboard_log)
            print(f"Loaded model from {checkpoint_load_path}")
        else:
            print(f"Checkpoint load path {checkpoint_load_path} does not exist. Starting training from scratch.")

    max_loops = learn_loop
    while learn_loop > 0:
        # Train the model
        model.learn(n_timesteps, reset_num_timesteps=False, tb_log_name=training_id)

        # Save the model
        if checkpoint_save_path is not None:
            model_id = max_loops - learn_loop + 1
            model.save(os.path.join(checkpoint_save_path, f"save_{model_id}.zip"))
            print(f"Saved model to {checkpoint_save_path}")

        learn_loop -= 1

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")