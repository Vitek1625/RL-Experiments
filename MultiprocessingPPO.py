import gymnasium as gym
import minigrid
import numpy as np

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from InstantiateEnvs import make_env

if __name__ == "__main__":
    env_id = "MiniGrid-Empty-Random-6x6-v0"
    flat_obs = True
    # env_id = "CartPole-v1"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i, path=f"results/{env_id}", filename="monitor", flat_obs=flat_obs) for i in range(num_cpu)])

    # print(env.observation_space)
    # exit()

    model = PPO("MlpPolicy", env, verbose=1)

    eval_env = make_env(env_id, 0, path=f"results/{env_id}", filename="monitor", monitor=False, flat_obs=flat_obs)()

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

    n_timesteps = 50_000

    # Multiprocessed RL Training
    model.learn(n_timesteps)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")