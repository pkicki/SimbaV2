from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax.lax import convert_element_type
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.hyper_sac.hyper_sac_update import l2normalize
from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.layers import MLPBlock, ResidualBlock
from scale_rl.networks.policies import NormalTanhPolicy
from scale_rl.networks.projection_utils import project_to_hypersphere
from scale_rl.networks.utils import he_normal_init, orthogonal_init

tfd = tfp.distributions
tfb = tfp.bijectors


class Scale(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    """
    init: value one would like to multiply to the input x.
    scale: value to control the scale of the gradient.
    """

    def setup(self):
        self.scaler = self.param(
            "scaler",
            nn.initializers.constant(1.0 * self.scale),
            self.dim,
        )
        self.forward_scaler = self.init / self.scale

    def __call__(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int
    dtype: Any
    use_scaler: bool
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=orthogonal_init(scale=1.0, axis=0),
            use_bias=self.use_bias,
            dtype=self.dtype,
        )(x)
        if self.use_scaler:
            x = Scale(self.hidden_dim, init=self.scaler_init, scale=self.scaler_scale)(
                x
            )

        return x


class HyperFeedForward(nn.Module):
    hidden_dim: int
    out_dim: int
    dtype: Any
    eps: float = 1e-5

    def setup(self):
        self.w1 = HyperDense(self.hidden_dim, use_scaler=True, dtype=self.dtype)
        self.w2 = HyperDense(self.out_dim, use_scaler=False, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        # note: when using relu,
        # it should be `x = nn.relu(x) + eps` to prevent zero vector.
        x = nn.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperResidualBlock(nn.Module):
    hidden_dim: int
    alpha_init: float
    alpha_scale: float
    dtype: Any

    def setup(self):
        self.ff = HyperFeedForward(
            hidden_dim=self.hidden_dim * 4, out_dim=self.hidden_dim, dtype=self.dtype
        )
        self.scaler = Scale(
            self.hidden_dim, init=self.alpha_init, scale=self.alpha_scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.ff(x)
        x = residual + self.scaler(x - residual)
        x = l2normalize(x, axis=-1)
        return x


class HyperEncoder(nn.Module):
    num_blocks: int
    hidden_dim: int
    input_projection_type: str
    input_projection_constant: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = project_to_hypersphere(
            x=x,
            projection_type=self.input_projection_type,
            constant=self.input_projection_constant,
        )
        x = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(x)
        x = l2normalize(x, axis=-1)

        for _ in range(self.num_blocks):
            x = HyperResidualBlock(
                hidden_dim=self.hidden_dim,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                dtype=self.dtype,
            )(x)

        return x


class HyperNormalTanhPolicy(nn.Module):
    action_dim: int
    kernel_init_scale: float = 1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    hidden_dim: int = 128
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        means = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(inputs)

        log_stds = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(inputs)

        means = HyperDense(
            hidden_dim=self.action_dim,
            dtype=self.dtype,
            use_scaler=False,
        )(means)

        log_stds = HyperDense(
            hidden_dim=self.action_dim,
            dtype=self.dtype,
            use_scaler=False,
        )(log_stds)

        mean_bias = self.param("mean_bias", nn.initializers.zeros, (self.action_dim,))
        log_std_bias = self.param(
            "log_std_bias", nn.initializers.zeros, (self.action_dim,)
        )

        means = means + mean_bias
        log_stds = log_stds + log_std_bias

        log_stds = convert_element_type(log_stds, jnp.float32)

        # suggested by Ilya for stability
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + nn.tanh(log_stds)
        )

        # N(mu, exp(log_sigma))
        dist = tfd.MultivariateNormalDiag(
            loc=convert_element_type(means, jnp.float32),
            scale_diag=jnp.exp(log_stds) * temperature,
        )

        # tanh(N(mu, sigma))
        dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())

        return dist


class HyperCritic(nn.Module):
    kernel_init_scale: float = 1.0
    hidden_dim: int = 512
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        value = inputs
        value = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(value)

        value = HyperDense(
            hidden_dim=1,
            dtype=self.dtype,
            use_scaler=False,
        )(value)

        bias = self.param("value_bias", nn.initializers.zeros, (1,))
        value = value + bias

        return value


class HyperSACActor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    input_projection_type: str
    input_projection_constant: float
    alpha_init: float
    alpha_scale: float
    output_hidden_dim: int
    dtype: Any

    def setup(self):
        self.encoder = HyperEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )
        self.predictor = HyperNormalTanhPolicy(
            action_dim=self.action_dim,
            hidden_dim=self.output_hidden_dim,
            dtype=self.dtype,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        observations = convert_element_type(observations, self.dtype)
        z = self.encoder(observations)
        dist = self.predictor(z, temperature)
        return dist


class HyperSACCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    input_projection_type: str
    input_projection_constant: float
    alpha_init: float
    alpha_scale: float
    output_hidden_dim: int
    dtype: Any

    def setup(self):
        self.encoder = HyperEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )

        self.predictor = HyperCritic(
            hidden_dim=self.output_hidden_dim,
            dtype=self.dtype,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, actions), axis=1)
        inputs = convert_element_type(inputs, self.dtype)
        z = self.encoder(inputs)
        q = self.predictor(z)
        return q


class HyperSACClippedDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    num_blocks: int
    hidden_dim: int
    input_projection_type: str
    input_projection_constant: float
    alpha_init: float
    alpha_scale: float
    output_hidden_dim: int
    dtype: Any

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            HyperSACCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            output_hidden_dim=self.output_hidden_dim,
            dtype=self.dtype,
        )(observations, actions)

        return qs


class HyperSACTemperature(nn.Module):
    initial_value: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            name="log_temp",
            init_fn=lambda key: jnp.full(
                shape=(), fill_value=jnp.log(self.initial_value)
            ),
        )
        return jnp.exp(log_temp)
