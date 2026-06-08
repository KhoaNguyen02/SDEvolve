import copy

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr

from .base_evaluator import BaseEvaluator


class ODEEvaluator(BaseEvaluator):
    def __init__(self, env, state_size, dt0, feedback_fn, solver=diffrax.Euler(), max_steps=16**4, stepsize_controller=diffrax.ConstantStepSize()):
        super().__init__(env, state_size, dt0)
        self.feedback_fn = feedback_fn
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller
        self.eta = 1.0

    def __call__(self, candidate, data, tree_evaluator):
        _, _, _, _, fitness, _, _ = jax.vmap(
            self.evaluate_trajectory,
            in_axes=[None, 0, None, 0, 0, 0, None]
        )(candidate, *data, tree_evaluator)
        return jnp.mean(fitness)

    def evaluate_trajectory(self, candidate, x0, ts, target, noise_key, params, tree_evaluator):
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        state_equation = candidate[:self.state_size]
        readout_equation = candidate[self.state_size:self.state_size + env.n_control_inputs]

        process_noise_key, obs_noise_key = jr.split(noise_key, 2)

        saveat = diffrax.SaveAt(ts=ts)
        targets = diffrax.LinearInterpolation(ts, target.squeeze(-1))

        a0 = jnp.zeros(self.state_size)
        rhat0 = jnp.zeros(1)
        _x0 = jnp.concatenate([x0, a0, rhat0])

        brownian_motion = diffrax.UnsafeBrownianPath(
            shape=(self.latent_size,),
            key=process_noise_key,
            levy_area=diffrax.SpaceTimeLevyArea,
        )

        system = diffrax.MultiTerm(
            diffrax.ODETerm(self._drift),
            diffrax.ControlTerm(self._diffusion, brownian_motion),
        )

        sol = diffrax.diffeqsolve(
            system, self.solver, ts[0], ts[-1], self.dt0, _x0,
            saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps,
            args=(env, state_equation, readout_equation, obs_noise_key, targets, tree_evaluator),
            stepsize_controller=self.stepsize_controller, throw=True
        )

        xs = sol.ys[:, :self.latent_size]
        activities = sol.ys[:, self.latent_size:self.latent_size + self.state_size]
        rhats = sol.ys[:, self.latent_size + self.state_size:]
        activities = jnp.tanh(activities)

        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))

        target_ts = targets.evaluate(ts)[:, None]
        feedbacks = jax.vmap(self.feedback_fn)(xs, target_ts)[:, None]
        rpes = feedbacks - rhats

        us = jax.vmap(
            lambda y, a, rpe: tree_evaluator(
                readout_equation,
                jnp.concatenate([y, a, rpe])
            ),
            in_axes=[0, 0, 0]
        )(ys, activities, rpes)
        us = jnp.tanh(us)

        fitness = env.fitness_function(xs, us, targets.evaluate(ts), ts)
        return xs, ys, us, activities, fitness, feedbacks, rpes

    def _drift(self, t, x_a_rhat, args):
        env, state_equation, readout_equation, obs_noise_key, target, tree_evaluator = args
        x = x_a_rhat[:self.latent_size]
        a = x_a_rhat[self.latent_size:self.latent_size + self.state_size]
        rhat = x_a_rhat[self.latent_size + self.state_size:]

        a = jnp.tanh(a)

        _, y = env.f_obs(obs_noise_key, (t, x))

        target_val = jnp.atleast_1d(target.evaluate(t))
        feedback = jnp.atleast_1d(self.feedback_fn(x, target_val))
        rpe = feedback - rhat

        u = tree_evaluator(
            readout_equation,
            jnp.concatenate([jnp.zeros(self.obs_size), a, rpe])
        )
        u = jnp.atleast_1d(jnp.tanh(u))

        dx = env.drift(t, x, u)
        da = tree_evaluator(
            state_equation, 
            jnp.concatenate([y, a, rpe])
        )
        drhat = self.eta * rpe

        return jnp.concatenate([dx, da, drhat])

    def _diffusion(self, t, x_a_rhat, args):
        env, state_equation, readout_equation, obs_noise_key, target, tree_evaluator = args
        x = x_a_rhat[:self.latent_size]

        Sigma_x = env.diffusion(t, x, jnp.array([0]))

        upper = Sigma_x
        middle = jnp.zeros((self.state_size, self.latent_size))
        bottom = jnp.zeros((1, self.latent_size))

        return jnp.concatenate([upper, middle, bottom], axis=0)