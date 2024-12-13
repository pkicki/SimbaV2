import gymnasium as gym

from scale_rl.envs.wrappers import DoNotTerminate

MUJOCO_ALL = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]


def make_mujoco_env(
    env_name: str,
    seed: int,
) -> gym.Env:
    env = gym.make(env_name, render_mode="rgb_array")

    return env
