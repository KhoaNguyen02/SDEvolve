import jax
import jax.numpy as jnp
import jax.random as jrandom

from .base_environment import EnvironmentBase


class HarmonicOscillator(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs=2):
        self.n_dim = 1
        self.n_var = 2
        self.n_control_inputs = 1
        self.n_targets = 1
        self.mu0 = jnp.zeros(self.n_var)
        self.P0 = jnp.eye(self.n_var) * jnp.array([2.0, 1.0])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.q = self.r = 0.5
        self.Q = jnp.array([[self.q, 0], [0, 0]])
        self.R = jnp.array([[self.r]])

    def sample_init_states(self, batch_size, ts, key, n_jumps):
        init_key, target_key, jump_key = jrandom.split(key, 3)
        x0 = self.mu0 + jrandom.normal(init_key, shape=(batch_size, self.n_var)) @ self.P0
        base_target = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=-3.0, maxval=3.0)

        if n_jumps == 0:
            targets = jnp.repeat(base_target[:, None, :], len(ts), axis=1)
        else:
            interval = len(ts) // n_jumps + len(ts) % n_jumps
            mask = jnp.arange(len(ts)) % interval == 0
            jumps = jrandom.uniform(jump_key, shape=(batch_size, len(ts)), minval=-2.0, maxval=2.0)
            target = base_target + jnp.cumsum(jumps * mask, axis=1)
            targets = target[:, :, None]
        return x0, targets

    def sample_params(self, batch_size, mode, ts, key):
        omega_key, zeta_key, args_key = jrandom.split(key, 3)
        if mode == "constant":
            omegas = jnp.ones((batch_size))
            zetas = jnp.zeros((batch_size))
        elif mode == "different":
            omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.0, maxval=2.0)
            zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.5)
        elif mode == "changing":
            decay_factors = jrandom.uniform(args_key, shape=(batch_size, 2), minval=0.98, maxval=1.02)
            init_omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.5, maxval=1.5)
            init_zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.0)
            omegas = jax.vmap(lambda o, d, t: o * (d ** t), in_axes=[0, 0, None])(init_omegas, decay_factors[:, 0], ts)
            zetas = jax.vmap(lambda z, d, t: z * (d ** t), in_axes=[0, 0, None])(init_zetas, decay_factors[:, 1], ts)
        return omegas, zetas

    def initialize_parameters(self, params, ts):
        omega, zeta = params
        self.A = jnp.array([[0, 1], [-omega, -zeta]])
        self.b = jnp.array([[0.0, 1.0]]).T
        self.G = jnp.array([[0, 0], [0, 1]])
        self.V = self.process_noise * self.G
        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise * jnp.eye(self.n_obs)

    def drift(self, t, state, args):
        return self.A @ state + self.b @ args

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, target, ts):
        x_d = jnp.stack([target, jnp.zeros_like(ts)], axis=1)
        u_d = jax.vmap(lambda _x_d: -jnp.linalg.pinv(self.b) @ self.A @ _x_d)(x_d)
        costs = jax.vmap(lambda _state, _u, _x_d, _u_d: 
            (_state - _x_d).T @ self.Q @ (_state - _x_d) + (_u - _u_d) @ self.R @ (_u - _u_d)
        )(state, control, x_d, u_d)
        return jnp.mean(costs)

    def cond_fn_nan(self, t, y, args, **kwargs):
        return jnp.where(jnp.any(jnp.isinf(y) + jnp.isnan(y)), -1.0, 1.0)
