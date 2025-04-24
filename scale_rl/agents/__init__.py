import gymnasium as gym
from typing import TypeVar
from omegaconf import OmegaConf
from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.wrappers import ObservationNormalizer, RewardNormalizer

Config = TypeVar('Config')


def create_agent(
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    cfg: Config,
) -> BaseAgent:

    cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
    agent_type = cfg.pop('agent_type')

    if agent_type == 'random':
        from scale_rl.agents.random_agent import RandomAgent
        agent = RandomAgent(observation_space, action_space, cfg)

    elif agent_type == 'simba':
        from scale_rl.agents.simba.simba_agent import SimbaAgent
        agent = SimbaAgent(observation_space, action_space, cfg)

    elif agent_type == 'simbaV2':
        from scale_rl.agents.simbaV2.simbaV2_agent import SimbaV2Agent
        agent = SimbaV2Agent(observation_space, action_space, cfg)

    elif agent_type == 'simbaV2lp':
        from scale_rl.agents.simbaV2.low_pass import SimbaV2LPAgent
        agent = SimbaV2LPAgent(observation_space, action_space, cfg)

    else:
        raise NotImplementedError

    # observation and reward normalization wrappers
    if cfg['normalize_observation']:
        agent = ObservationNormalizer(
            agent, 
            load_rms=cfg['load_observation_normalizer']
        )
    if cfg['normalize_reward']:
        agent = RewardNormalizer(
            agent, 
            gamma=cfg['gamma'], 
            g_max=cfg['normalized_g_max'],
            load_rms=cfg['load_reward_normalizer']
        )

    return agent