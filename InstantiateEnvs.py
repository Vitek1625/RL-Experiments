import gymnasium as gym

from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper
import os

def make_env(env_id: str, rank: int, seed: int = 0, path:str = "results", filename: str = "monitor", file_ext: str = ".csv",
              monitor: bool = True, flat_obs: bool = False, img_obs: bool = False) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param path: (str) the directory to save the monitoring results
    :param filename: (str) the file to save the monitoring results
    :param file_ext: (str) the extension of the monitoring file
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)

        if flat_obs:
            env = FlatObsWrapper(env)
    
        if img_obs:
            env = ImgObsWrapper(env)
            
        if monitor:
            env = Monitor(env, filename=os.path.join(path, filename + f"_{seed + rank}"))

        env.reset(seed=seed + rank)
        return env

    if not os.path.isdir(path):
        os.makedirs(path)

    set_random_seed(seed)
    return _init