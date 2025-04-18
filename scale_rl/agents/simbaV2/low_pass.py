import jax.numpy as jnp
from scipy.signal import butter
import jax
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.simbaV2.simbaV2_layer import HyperEmbedder, HyperLERPBlock, HyperNormalTanhPolicy
tfd = tfp.distributions
tfb = tfp.bijectors

class LowPassNoiseDist(tfd.Distribution):
    def __init__(self, cutoff=1.0, order=1, sampling_freq=20., seq_len=100., key=None, loc=None, scale_diag=None, validate_args=False, allow_nan_stats=True, name="MultivariateNormalDiag"):
        parameters = dict(locals())
        #with tfp.util.deferred_dependencies.defer_dependencies():
        self._loc = jnp.zeros_like(scale_diag) if loc is None else jnp.asarray(loc)
        self._scale_diag = jnp.asarray(scale_diag)

        self.cutoff = cutoff
        self.order = order
        self.sampling_freq = sampling_freq
        self.seq_len = seq_len
        
        #self.gen = LowPassNoiseProcess(cutoff=self.cutoff, order=self.order, sampling_freq=self.sampling_freq,
        #                               size=(self.seq_len, scale_diag.shape[-1]), key=key)

        if self._loc.shape != self._scale_diag.shape:
            raise ValueError(f"Shape mismatch: loc {self._loc.shape} vs scale_diag {self._scale_diag.shape}")
        
        self._batch_shape_ = jax.lax.broadcast_shapes(self._loc.shape[:-1], self._scale_diag.shape[:-1])
        self._event_shape_ = self._loc.shape[-1:]

        self.b, self.a = butter(self.order, self.cutoff, fs=self.sampling_freq)
        self.b_jax = jnp.array(self.b)
        self.a_jax = jnp.array(self.a)

        self.x_hist = jnp.zeros((self.order+1, self._loc.shape[-1]))
        self.y_hist = jnp.zeros((self.order, self._loc.shape[-1]))
        
        super().__init__(
            dtype=self._loc.dtype,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @property
    def loc(self):
        return self._loc

    @property
    def scale_diag(self):
        return self._scale_diag

    def _batch_shape(self):
        return self._batch_shape_

    def _event_shape(self):
        return self._event_shape_

    def _sample_n(self, sample_shape, seed):
        key = jax.random.split(seed)[0]
        if type(sample_shape) is not tuple:
            sample_shape = (sample_shape,)
        eps = jax.random.normal(key, shape=sample_shape + self._batch_shape_ + self._event_shape_)
        if self._loc.shape[0] == 1:
            # filter the signal
            self.x_hist = jnp.roll(self.x_hist, shift=1, axis=0)
            self.x_hist = self.x_hist.at[0].set(eps[0, 0])
            y = jnp.dot(self.b_jax, self.x_hist) - jnp.dot(self.a_jax[1:], self.y_hist)
            self.y_hist = jnp.roll(self.y_hist, shift=1, axis=0)
            self.y_hist = self.y_hist.at[0].set(y)
            eps = y[None, None]

        return self._loc + eps * self._scale_diag

    def _log_prob(self, value):
        var = jnp.square(self._scale_diag)
        log_scale = jnp.log(self._scale_diag)
        return -0.5 * (jnp.square(value - self._loc) / var + 2. * log_scale + jnp.log(2. * jnp.pi)).sum(axis=-1)

    def _mean(self):
        return self._loc

    def _stddev(self):
        return self._scale_diag

    def set_mean_and_scale_diag(self, mean, scale_diag):
        self._loc = mean
        self._scale_diag = scale_diag
        self._batch_shape_ = jax.lax.broadcast_shapes(self._loc.shape[:-1], self._scale_diag.shape[:-1])

    def _entropy(self):
        return jnp.sum(
            jnp.log(self._scale_diag * jnp.sqrt(2. * jnp.pi * jnp.e)),
            axis=-1
        )

    def mode(self):
        return self._loc

    def __repr__(self) -> str:
        return f"LowPassNoiseDist(cutoff={self.cutoff}, order={self.order}, sampling_freq={self.sampling_freq}, seq_len={self.seq_len})"

        
class SimbaV2LPActor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    cutoff: float
    order: int
    sampling_freq: float
    seq_len: int

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.predictor = HyperNormalTanhPolicy(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )
        self.dist = LowPassNoiseDist(
            cutoff=self.cutoff,
            order=self.order,
            sampling_freq=self.sampling_freq,
            seq_len=self.seq_len,
            loc=jnp.zeros((self.action_dim,)),
            scale_diag=jnp.ones((self.action_dim,)),
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        x = observations
        y = self.embedder(x)
        z = self.encoder(y)
        dist, info = self.predictor(z, temperature)
        self.dist.set_mean_and_scale_diag(dist.distribution.mean(), dist.distribution.stddev())
        dist = tfd.TransformedDistribution(distribution=self.dist, bijector=tfb.Tanh())

        return dist, info