import copy

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
from .base_evaluator import BaseEvaluator


class SHOEvaluator(BaseEvaluator):
    def __init__(self, env, state_size, dt0, learn_condition, feedback_fn, sign_estimation, 
                solver=diffrax.Euler(), max_steps=16**4, stepsize_controller=diffrax.ConstantStepSize()):
        assert learn_condition in [1, 2, 3], "learn_condition must be 1, 2, or 3"
        assert learn_condition != 2 or feedback_fn is not None, "Feedback function required for condition 2"
        super().__init__(env, state_size, dt0)
        self.learn_condition = learn_condition
        self.feedback_fn = feedback_fn
        self.sign_estimation = sign_estimation
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller
        self.eps = 0.01

    def __call__(self, candidate, data, tree_evaluator):
        _, _, _, _, fitness, fbs = jax.vmap(
            self.evaluate_trajectory,
            in_axes=[None, 0, None, 0, 0, 0, None]
        )(candidate, *data, tree_evaluator)
        return jnp.mean(fitness)

    def evaluate_trajectory(self, candidate, x0, ts, target, noise_key, params, tree_evaluator):
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        state_equation = candidate[:self.state_size]
        readout = candidate[self.state_size:]

        saveat = diffrax.SaveAt(ts=ts)
        process_noise_key, obs_noise_key = jr.split(noise_key, 2)

        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])
        targets = diffrax.LinearInterpolation(ts, target.squeeze(-1))

        brownian_motion = diffrax.UnsafeBrownianPath(
            shape=(self.latent_size,), 
            key=process_noise_key,
            levy_area=diffrax.SpaceTimeLevyArea
        )

        system = diffrax.MultiTerm(
            diffrax.ODETerm(self._drift), 
            diffrax.ControlTerm(self._diffusion, brownian_motion)
        )

        sol = diffrax.diffeqsolve(
            system, self.solver, ts[0], ts[-1], self.dt0, _x0,
            saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps,
            args=(env, state_equation, readout, obs_noise_key, targets, tree_evaluator),
            stepsize_controller=self.stepsize_controller, throw=False
        )

        xs = sol.ys[:, :self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:, self.latent_size:]

        if self.learn_condition == 1:
            ys_ = jnp.zeros_like(ys)
            feedbacks = targets.evaluate(ts)[:, None]
        elif self.learn_condition == 2:
            ys_ = jnp.zeros_like(ys)
            if self.sign_estimation:
                keys = jr.split(obs_noise_key, len(ts))
                feedbacks = jax.vmap(
                    lambda t, x, k: self._get_feedback(x, jnp.squeeze(targets.evaluate(t)), k)
                )(ts, xs, keys)
            else:
                feedbacks = jax.vmap(self.feedback_fn)(xs, targets.evaluate(ts)[:, None])
        else:
            feedbacks = jnp.zeros_like(targets.evaluate(ts)[:, None])
            ys_ = jnp.zeros_like(ys)

        us = jax.vmap(lambda y, a, tar: tree_evaluator(
            readout, jnp.concatenate([y, a, jnp.zeros(self.control_size), tar])), in_axes=[0, 0, 0]
        )(ys_, activities, feedbacks)
        fitness = env.fitness_function(xs, us, targets.evaluate(ts), ts)
        return xs, ys, us, activities, fitness, feedbacks

    def _drift(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]
        target_t = jnp.atleast_1d(target.evaluate(t))

        _, y = env.f_obs(obs_noise_key, (t, x))

        if self.learn_condition == 1:
            pass
        elif self.learn_condition == 2:
            if self.sign_estimation:
                feedback_t = self._get_feedback(x, target_t, jr.fold_in(obs_noise_key, jnp.int16(jnp.round(t / self.dt0))))
                target_t = jnp.atleast_1d(feedback_t)
            else:
                feedback_t = jnp.atleast_1d(self.feedback_fn(x, target_t))
                target_t = feedback_t
        else:
            _, y = env.f_obs(obs_noise_key, (t, x), noisy=False)
            target_t = jnp.zeros_like(target_t)

        u = tree_evaluator(
            readout,
            jnp.concatenate([jnp.zeros(self.obs_size), a, jnp.zeros(self.control_size), target_t])
        )

        # Apply control to system and get system change
        dx = env.drift(t, x, u)
        da = tree_evaluator(
            state_equation,
            jnp.concatenate([y, a, u, target_t])
        )
        return jnp.concatenate([dx, da])

    def _diffusion(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]
        return jnp.concatenate(
            [env.diffusion(t, x, jnp.array([0])), jnp.zeros((self.state_size, self.latent_size))]
        )

    def _get_feedback(self, x, target, key):
        k = jnp.ones_like(x)
        delta = jr.uniform(key, minval=0.1, maxval=1.0, shape=x.shape) * self.eps
        e_p = self.feedback_fn(x + delta * k, target)
        e_m = self.feedback_fn(x - delta * k, target)
        s = -jnp.sign(e_p - e_m)
        d = self.feedback_fn(x, target)
        return jnp.reshape(s * d, (1,))