import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr

from environments.harmonic_oscillator import HarmonicOscillator
from environments.duffing_oscillator import DuffingOscillator
from evaluators.ode_evaluator import ODEEvaluator
from evaluators.sde_evaluator import SDEEvaluator
from kozax.genetic_programming import GeneticProgramming


def get_data(key, env, batch_size, dt, T, lambd, jump_size):
    """Generate data from dynamical system

    Parameters
    ----------
    key : jax.random.PRNGKey
        jax random key for reproducibility
    env : object
        control task (e.g. harmonic oscillator, duffing oscillator)
    batch_size : int
        number of trajectories to generate
    dt : float
        simulation time step
    T : float
        total simulation time
    lambd : float
        intensity of the target jump (for changing target experiments)
    jump_size : float
        size of the target jump (for changing target experiments)

    Returns
    -------
    x0 : jnp.ndarray
        initial states of shape (batch_size, state_size)
    ts : jnp.ndarray
        time steps of shape (num_time_steps,)
    targets : jnp.ndarray
        target trajectories of shape (batch_size, num_time_steps, state_size)
    noise_keys : jnp.ndarray
        random keys for noise generation, of shape (batch_size, 2)
    params : jnp.ndarray
        environment parameters of shape (batch_size, num_params)
    """
    init_key, noise_key, param_key = jr.split(key, 3)
    ts = jnp.arange(0, T, dt)
    params = env.sample_params(batch_size, ts, param_key)
    x0, targets = env.sample_init_states(batch_size, ts, init_key, lambd, jump_size, params)
    noise_keys = jr.split(noise_key, batch_size)
    return x0, ts, targets, noise_keys, params


def setup(program, env_name, n_obs, feedback_fn, process_noise=0.1, obs_noise=0.05, gamma=0.0):
    """Setup environment, strategy, and model for validation

    Parameters
    ----------
    program : str
        model name (e.g. "GP-ODE", "GP-SDE")
    env_name : str
        environment name (e.g. "HO", "DO")
    n_obs : int
        number of observations (1 for partial observability, 2 for full observability)
    feedback_fn : callable
        reward signal
    process_noise : float, optional
        environment process noise level, by default 0.1
    obs_noise : float, optional
        environment observation noise level, by default 0.05
    gamma : float, optional
        maximum diffusion coefficient for GP-SDE, by default 0.0 (no diffusion)

    Returns
    -------
    env : object
        initialized environment
    strategy : GeneticProgramming
        initialized GP strategy
    fitness_function : ODEEvaluator or SDEEvaluator
        initialized model for evaluating candidate policies
    """
    state_size = 2
    dt0 = 0.05
    max_steps = 1000
 
    if env_name == "HO":
        env = HarmonicOscillator(process_noise=process_noise, obs_noise=obs_noise, n_obs=n_obs)
    elif env_name == "DO":
        env = DuffingOscillator(process_noise=process_noise, obs_noise=obs_noise, n_obs=n_obs)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    memory_variables = [f"y{i}" for i in range(env.n_obs)] + [f"a{i}" for i in range(state_size)] + ["rpe"]
    readout_variables = [f"a{i}" for i in range(state_size)] + ["rpe"]

    if program == "GP-ODE":
        layer_sizes = jnp.array([state_size, env.n_control_inputs])
        variable_list = [memory_variables, readout_variables]
        fitness_function = ODEEvaluator(env, state_size, dt0, feedback_fn, solver=diffrax.GeneralShARK(), max_steps=max_steps)
    elif program == "GP-SDE":
        layer_sizes = jnp.array([state_size, env.n_control_inputs, state_size])
        variable_list = [memory_variables, readout_variables, memory_variables]
        fitness_function = SDEEvaluator(env, state_size, dt0, feedback_fn, gamma=gamma, solver=diffrax.GeneralShARK(), max_steps=max_steps)
    else:
        raise ValueError(f"Unknown program: {program}")

    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
        ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1),
    ]
 
    strategy = GeneticProgramming(
        num_generations=1,
        population_size=100,
        fitness_function=fitness_function,
        operator_list=operator_list,
        variable_list=variable_list,
        layer_sizes=layer_sizes,
        num_populations=10,
        device_type=jax.devices()[0].platform
    )

    return env, strategy, fitness_function


def validate(program, strategy, fitness_function, candidate, test_data):
    """Validate candidate policies

    Parameters
    ----------
    program : str
        model name (e.g. "GP-ODE", "GP-SDE")
    strategy : GeneticProgramming
        initialized GP strategy
    fitness_function : ODEEvaluator or SDEEvaluator
        initialized model for evaluating candidate policies
    candidate : object
        candidate policy to validate
    test_data : object
        validation dataset

    Returns
    -------
    ts : jnp.ndarray
        simulation time steps of shape (num_time_steps,)
    xs : jnp.ndarray
        states of shape (num_time_steps, state_size)
    ys : jnp.ndarray
        observed states of shape (num_time_steps, n_obs)
    us : jnp.ndarray
        control inputs of shape (num_time_steps, n_control_inputs)
    activities : jnp.ndarray
        memory activities of shape (num_time_steps, num_neurons)
    targets : jnp.ndarray
        target values of shape (num_time_steps, state_size)
    rs : jnp.ndarray
        reward signals of shape (num_time_steps,)
    rpes : jnp.ndarray
        reward prediction errors of shape (num_time_steps,)
    fitness : jnp.ndarray
        fitness values of shape (num_time_steps,)
    diffusion_coefs : jnp.ndarray
        diffusion coefficients of shape (num_time_steps, state_size)
    """
    if program == "GP-SDE":
        xs, ys, us, activities, fitness, rs, rpes, diffusion_coefs = jax.vmap(
            fitness_function.evaluate_trajectory,
            in_axes=[None, 0, None, 0, 0, 0, None],
        )(candidate, *test_data, strategy.tree_evaluator)
    elif program == "GP-ODE":
        xs, ys, us, activities, fitness, rs, rpes = jax.vmap(
            fitness_function.evaluate_trajectory,
            in_axes=[None, 0, None, 0, 0, 0, None],
        )(candidate, *test_data, strategy.tree_evaluator)
        diffusion_coefs = jnp.zeros_like(xs)

    ts, targets = test_data[1], test_data[2]
    return ts, xs, ys, us, activities, targets, rs, rpes, fitness, diffusion_coefs
