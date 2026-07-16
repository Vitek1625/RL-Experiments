from MultiprocessingPPO import train_multiprocessing_ppo


if __name__ == "__main__":
    config = {
        "env": {
            "id": "MiniGrid-Empty-Random-6x6-v0",
            "num_cpu": 8,
            "filepath": "results/MiniGrid-Empty-Random-6x6-v0",
            "filename": "monitor",
        },
        "wrappers": {
                "flat_obs": True,
                "monitor": True,
        },
        "model": {
            "policy": "MlpPolicy",
            "verbose": 1,  # For more detailed output during training
            "learning_rate": 0.0003,  # Learning rate for the optimizer
            "n_steps": 2048,  # Number of steps to run for each environment per update
            "batch_size": 64,  # Minibatch size for each gradient update
            "n_epochs": 10,  # Number of epochs to perform for each update
            "gamma": 0.99,  # Discount factor
            "gae_lambda": 0.95,  # GAE lambda parameter
            "clip_range": 0.2,  # Clipping parameter for PPO
            "ent_coef": 0.0,  # Coefficient for the entropy term
        },
        "save": {
            "load": None,  # Path to load a pre-trained model (if any)
            "path": "results/MiniGrid-Empty-Random-6x6-v0/ppo_model",  # Path to save the trained model
            "filename": "ppo_model",  # Name of the model file
        },
        "learn": {
            "n_timesteps": 50_000,
            "eval_episodes": 10,  # Number of episodes for evaluation
            "eval_freq": 1000,  # Frequency of evaluation
        },
    }
    train_multiprocessing_ppo(config)