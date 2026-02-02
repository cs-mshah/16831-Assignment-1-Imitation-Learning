import os
import gym
import numpy as np
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from rob831.infrastructure import utils

EXPERT_POLICIES_DIR = "../policies/experts/"
ENVS = {
    "Ant-v2": "Ant.pkl",
    "Humanoid-v2": "Humanoid.pkl",
    "Walker2d-v2": "Walker2d.pkl",
    "Hopper-v2": "Hopper.pkl",
    "HalfCheetah-v2": "HalfCheetah.pkl",
}


def main():
    results = {}

    print(f"{'Environment':<20} | {'Mean Return':<15} | {'Std Return':<15}")
    print("-" * 56)

    for env_name, policy_file_name in ENVS.items():
        try:
            env = gym.make(env_name)
            env.seed(1)
        except Exception as e:
            print(f"Skipping {env_name}: {e}")
            continue

        policy = LoadedGaussianPolicy(os.path.join(EXPERT_POLICIES_DIR, policy_file_name))

        max_path_length = 1000

        print(f"Running {env_name}...")
        paths = utils.sample_n_trajectories(
            env, policy, ntraj=2, max_path_length=max_path_length
        )

        returns = [path["reward"].sum() for path in paths]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        results[env_name] = (mean_return, std_return)

        print(f"{env_name:<20} | {mean_return:<15.4f} | {std_return:<15.4f}")

        env.close()


if __name__ == "__main__":
    main()
