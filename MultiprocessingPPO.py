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

    env_id = main_config["env"]["id"]

    monitor_runs = main_config["wrappers"].get("monitor", False)
    flat_obs = main_config["wrappers"].get("flat_obs", False)
    img_obs = main_config["wrappers"].get("img_obs", False)
    num_cpu = main_config["env"].get("num_cpu", 1)

    result_filepath = main_config["env"].get("filepath", os.path.join("results", env_id))
    result_filename = main_config["env"].get("filename", "monitor")

    n_timesteps = main_config["learn"].get("n_timesteps", 50_000)
    eval_episodes = main_config["learn"].get("eval_episodes", 10)

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i, path=result_filepath, filename=result_filename, monitor=monitor_runs, flat_obs=flat_obs, img_obs=img_obs) for i in range(num_cpu)])

    # print(env.observation_space)
    # exit()

    model = PPO(env=env, **main_config.get("model", {}))

    eval_env = make_env(env_id, 0, monitor=False, flat_obs=flat_obs, img_obs=img_obs)()

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


    # Multiprocessed RL Training
    model.learn(n_timesteps)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")