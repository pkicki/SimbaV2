import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.hyper_sac.hyper_sac_network import (
    HyperSACActor,
    HyperSACClippedDoubleCritic,
    HyperSACCritic,
    HyperSACTemperature,
)
from scale_rl.agents.hyper_sac.hyper_sac_update import (
    l2normalize_network,
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)
from scale_rl.buffers.base_buffer import Batch
from scale_rl.networks.trainer import PRNGKey, Trainer

"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class HyperSACConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate_init: float
    actor_learning_rate_end: float
    actor_learning_rate_decay_rate: float
    actor_learning_rate_decay_step: int
    actor_input_projection_type: str
    actor_input_projection_constant: float
    actor_alpha_init: float
    actor_alpha_scale: float
    actor_output_hidden_dim: int

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate_init: float
    critic_learning_rate_end: float
    critic_learning_rate_decay_rate: float
    critic_learning_rate_decay_step: int
    critic_use_cdq: bool
    critic_input_projection_type: str
    critic_input_projection_constant: float
    critic_alpha_init: float
    critic_alpha_scale: float
    critic_output_hidden_dim: int

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float

    target_tau: float
    target_normalize_weight: bool
    gamma: float
    n_step: int

    mixed_precision: bool


@functools.partial(
    jax.jit,
    static_argnames=(
        "observation_dim",
        "action_dim",
        "cfg",
    ),
)
def _init_hyper_sac_networks(
    observation_dim: int,
    action_dim: int,
    cfg: HyperSACConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    compute_dtype = jnp.float16 if cfg.mixed_precision else jnp.float32

    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    actor = Trainer.create(
        network_def=HyperSACActor(
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            input_projection_type=cfg.actor_input_projection_type,
            input_projection_constant=cfg.actor_input_projection_constant,
            alpha_init=cfg.actor_alpha_init,
            alpha_scale=cfg.actor_alpha_scale,
            output_hidden_dim=cfg.actor_output_hidden_dim,
            dtype=compute_dtype,
        ),
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=cfg.actor_learning_rate_init,
                end_value=cfg.actor_learning_rate_end,
                transition_steps=cfg.actor_learning_rate_decay_step,
            ),
            weight_decay=0.0,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
    )

    if cfg.critic_use_cdq:
        critic_network_def = HyperSACClippedDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            input_projection_type=cfg.critic_input_projection_type,
            input_projection_constant=cfg.critic_input_projection_constant,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
            output_hidden_dim=cfg.critic_output_hidden_dim,
            dtype=compute_dtype,
        )
    else:
        critic_network_def = HyperSACCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            input_projection_type=cfg.critic_input_projection_type,
            input_projection_constant=cfg.critic_input_projection_constant,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
            output_hidden_dim=cfg.critic_output_hidden_dim,
            dtype=compute_dtype,
        )

    critic = Trainer.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=cfg.critic_learning_rate_init,
                end_value=cfg.critic_learning_rate_end,
                transition_steps=cfg.critic_learning_rate_decay_step,
            ),
            weight_decay=0.0,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
    )

    # we set target critic's parameters identical to critic by using same rng.
    target_network_def = critic_network_def
    target_critic = Trainer.create(
        network_def=target_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=None,
    )

    temperature = Trainer.create(
        network_def=HyperSACTemperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=optax.adamw(
            learning_rate=cfg.temp_learning_rate,
            weight_decay=0.0,
        ),
    )

    # l2-normalize the network after initialization
    actor = l2normalize_network(actor)
    critic = l2normalize_network(critic)
    target_critic = l2normalize_network(target_critic)

    return rng, actor, critic, target_critic, temperature


@jax.jit
def _sample_hyper_sac_actions(
    rng: PRNGKey,
    actor: Trainer,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist = actor(observations=observations, temperature=temperature)
    actions = dist.sample(seed=key)
    actions = jnp.clip(actions, -1.0, 1.0)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "target_tau",
        "target_normalize_weight",
        "temp_target_entropy",
    ),
)
def _update_hyper_sac_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    target_tau: float,
    target_normalize_weight: bool,
    temp_target_entropy: float,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        critic_use_cdq=critic_use_cdq,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
        normalize_weight=target_normalize_weight,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return (rng, new_actor, new_critic, new_target_critic, new_temperature, info)


class HyperSACAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: HyperSACConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(HyperSACAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        # map dictionary to dataclass
        self._cfg = HyperSACConfig(**cfg)

        self._init_network()

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_hyper_sac_networks(self._observation_dim, self._action_dim, self._cfg)

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        # current timestep observation is "next" observations from the previous timestep
        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = _sample_hyper_sac_actions(
            self._rng, self._actor, observations, temperature
        )
        actions = np.array(actions)

        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_hyper_sac_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            target_tau=self._cfg.target_tau,
            target_normalize_weight=self._cfg.target_normalize_weight,
            temp_target_entropy=self._cfg.temp_target_entropy,
        )

        for key, value in update_info.items():
            update_info[key] = float(value)

        return update_info
