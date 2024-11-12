from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from scale_rl.buffers import Batch
from scale_rl.networks.projection_utils import l2normalize
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.networks.utils import tree_map_until_match, tree_norm


def l2normalize_layer(tree):
    """
    apply l2-normalization to the all leaf nodes
    """
    if len(tree["kernel"].shape) == 2:
        axis = 0
    elif len(tree["kernel"].shape) == 3:
        axis = 1
    else:
        raise ValueError
    return jax.tree.map(f=lambda x: l2normalize(x, axis=axis), tree=tree)


def l2normalize_network(
    network: Trainer,
    regex: str = "hyper_dense",
) -> Trainer:
    params = network.params
    new_params = tree_map_until_match(
        f=lambda x: l2normalize_layer(x), tree=params, target_re=regex, keep_values=True
    )
    network = network.replace(params=new_params)
    return network


def update_actor(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,  # SACDoubleCritic
    temperature: Trainer,
    batch: Batch,
    critic_use_cdq: bool,
) -> Tuple[Trainer, Dict[str, float]]:
    def actor_loss_fn(
        actor_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist = actor.apply(
            variables={"params": actor_params},
            observations=batch["observation"],
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        if critic_use_cdq:
            q1, q2 = critic(observations=batch["observation"], actions=actions)
            q = jnp.minimum(q1, q2).reshape(-1)  # (n, 1) -> (n, )
        else:
            q = critic(observations=batch["observation"], actions=actions)
            q = q.reshape(-1)  # (n, 1) -> (n, )

        actor_loss = (log_probs * temperature() - q).mean()
        actor_info = {
            "actor_loss": actor_loss,
            "entropy": -log_probs.mean(),  # not exactly entropy, just calculating randomness
            "actor_action": jnp.mean(jnp.abs(actions)),
            "actor_pnorm": tree_norm(actor_params),
        }

        return actor_loss, actor_info

    actor, info = actor.apply_gradient(actor_loss_fn)
    info["actor_gnorm"] = info.pop("grad_norm")
    actor = l2normalize_network(actor)

    return actor, info


def update_critic(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
) -> Tuple[Trainer, Dict[str, float]]:
    # compute the target q-value
    next_dist = actor(observations=batch["next_observation"])
    next_actions = next_dist.sample(seed=key)
    next_log_probs = next_dist.log_prob(next_actions)
    if critic_use_cdq:
        next_q1, next_q2 = target_critic(
            observations=batch["next_observation"], actions=next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2).reshape(-1)
    else:
        next_q = target_critic(
            observations=batch["next_observation"],
            actions=next_actions,
        ).reshape(-1)

    # compute the td-target, incorporating the n-step accumulated reward
    # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
    target_reward_q = (
        batch["reward"] + (gamma**n_step) * (1 - batch["terminated"]) * next_q
    )
    target_entropy_q = (
        -(gamma**n_step) * (1 - batch["terminated"]) * temperature() * next_log_probs
    )
    target_q = target_reward_q + target_entropy_q

    def critic_loss_fn(
        critic_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # compute predicted q-value
        if critic_use_cdq:
            pred_q1, pred_q2 = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            pred_q1 = pred_q1.reshape(-1)
            pred_q2 = pred_q2.reshape(-1)

            # compute mse loss
            critic_loss = ((pred_q1 - target_q) ** 2 + (pred_q2 - target_q) ** 2).mean()
        else:
            pred_q = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            ).reshape(-1)
            pred_q1 = pred_q2 = pred_q

            # compute mse loss
            critic_loss = ((pred_q - target_q) ** 2).mean()

        critic_info = {
            "critic_loss": critic_loss,
            "q1_min": pred_q1.min(),
            "q1_mean": pred_q1.mean(),
            "q1_max": pred_q1.max(),
            "q2_mean": pred_q2.mean(),
            "rew_min": batch["reward"].min(),
            "rew_mean": batch["reward"].mean(),
            "rew_max": batch["reward"].max(),
            "critic_pnorm": tree_norm(critic_params),
        }

        return critic_loss, critic_info

    critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    info["critic_target_reward_q_min"] = target_reward_q.min()
    info["critic_target_reward_q_mean"] = target_reward_q.mean()
    info["critic_target_reward_q_max"] = target_reward_q.max()
    info["critic_target_entropy_q_min"] = target_entropy_q.min()
    info["critic_target_entropy_q_mean"] = target_entropy_q.mean()
    info["critic_target_entropy_q_max"] = target_entropy_q.max()
    critic = l2normalize_network(critic)

    return critic, info


def update_target_network(
    network: Trainer,  # SACDoubleCritic
    target_network: Trainer,
    target_tau: float,
    normalize_weight: bool = False,
) -> Tuple[Trainer, Dict[str, float]]:
    new_target_params = jax.tree_map(
        lambda p, tp: p * target_tau + tp * (1 - target_tau),
        network.params,
        target_network.params,
    )

    target_network = target_network.replace(params=new_target_params)
    info = {}
    if normalize_weight:
        target_network = l2normalize_network(target_network)

    return target_network, info


def update_temperature(
    temperature: Trainer, entropy: float, target_entropy: float
) -> Tuple[Trainer, Dict[str, float]]:
    def temperature_loss_fn(
        temperature_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        temperature_value = temperature.apply({"params": temperature_params})
        temperature_loss = temperature_value * (entropy - target_entropy).mean()
        temperature_info = {
            "temperature": temperature_value,
            "temperature_loss": temperature_loss,
        }

        return temperature_loss, temperature_info

    temperature, info = temperature.apply_gradient(temperature_loss_fn)
    info["temperature_gnorm"] = info.pop("grad_norm")

    return temperature, info
