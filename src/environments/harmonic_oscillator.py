import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from .base_environment import EnvironmentBase


class HarmonicOscillator(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs=2):
        self.env_name = "HO"
        self.n_dim = 1
        self.n_var = 2
        self.n_control_inputs = 1
        self.n_targets = 1
        self.mu0 = jnp.zeros(self.n_var)
        self.P0 = jnp.eye(self.n_var) * jnp.array([2.0, 1.0])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.Q = jnp.diag(jnp.array([0.5, 0.0]))
        self.R = jnp.array([[0.3]])

    def sample_init_states(self, batch_size, ts, key, lambd, jump_size, params):
        init_key, target_key, jump_key = jrandom.split(key, 3)
        x0 = self.mu0 + jrandom.normal(init_key, shape=(batch_size, self.n_var)) @ self.P0
        base_target = jrandom.uniform(target_key, shape=(batch_size,), minval=-2.0, maxval=2.0)

        if lambd == 0.0:
            targets = jnp.repeat(base_target[:, None, None], len(ts), axis=1)
        else:
            dt = ts[1] - ts[0]
            dN = jrandom.poisson(jump_key, lam=lambd * dt, shape=(batch_size, len(ts)))
            dY = jnp.sqrt(dN) * jump_size * jrandom.normal(jump_key, shape=(batch_size, len(ts)))
            targets = base_target[:, None] + jnp.cumsum(dY, axis=1)
            targets = targets[:, :, None]
        return x0, targets

    def sample_params(self, batch_size, ts, key):
        beta_key, delta_key = jrandom.split(key, 2)
        betas = jrandom.uniform(beta_key, shape=(batch_size,), minval=0.0, maxval=1.0)[:, None] * jnp.ones((batch_size, ts.shape[0]))
        deltas = jrandom.uniform(delta_key, shape=(batch_size,), minval=0.0, maxval=1.0)[:, None] * jnp.ones((batch_size, ts.shape[0]))
        return betas, deltas

    def initialize_parameters(self, params, ts):
        beta, delta = params
        A = jax.vmap(lambda b, d: jnp.array([[0, 1], [-b, -d]]))(beta, delta)
        self.A = diffrax.LinearInterpolation(ts, A)
        self.b = jnp.array([[0.0, 1.0]]).T
        self.G = jnp.array([[0, 0], [0, 1]])
        self.V = self.process_noise*self.G
        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise * jnp.eye(self.n_obs)

    def drift(self, t, state, args):
        return self.A.evaluate(t) @ state + self.b @ args

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, target, ts):
        x_d = jnp.stack([target, jnp.zeros_like(ts)], axis=1)
        u_d = -jax.vmap(lambda t, xd: (jnp.linalg.pinv(self.b) @ (self.A.evaluate(t) @ xd)))(ts, x_d)
        costs = jax.vmap(lambda _state, _u, _x_d, _u_d:
            (_state - _x_d).T @ self.Q @ (_state - _x_d) + (_u - _u_d).T @ self.R @ (_u - _u_d)
        )(state, control, x_d, u_d)
        return jnp.mean(costs)

    def cond_fn_nan(self, t, y, args, **kwargs):
        return jnp.where(jnp.any(jnp.isinf(y) + jnp.isnan(y)), -1.0, 1.0)