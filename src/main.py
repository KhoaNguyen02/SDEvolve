import os
import argparse
import shlex
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax

from environments.harmonic_oscillator import HarmonicOscillator
from environments.duffing_oscillator import DuffingOscillator
from evaluators.ode_evaluator import ODEEvaluator
from evaluators.sde_evaluator import SDEEvaluator
from kozax.genetic_programming import GeneticProgramming
from utils import get_data


def load_exp_info(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--process_noise", type=float, required=True)
    parser.add_argument("--obs_noise", type=float, required=True)
    parser.add_argument("--feedback_fn", type=str, default="None")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num_generations", type=int, default=0)
    parser.add_argument("--population_size", type=int, default=0)
    parser.add_argument("--num_populations", type=int, default=0)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--lambd", type=float, required=True)
    parser.add_argument("--jump_size", type=float, required=True)

    lines = Path(path).read_text().splitlines()
    clean = []
    for line in lines:
        line = line.split("#")[0].strip()
        if line:
            clean.append(line)

    return parser.parse_args(shlex.split(" ".join(clean)))


def run(program, env_name, exp_info, n_obs):
    # Load experiment config
    args = load_exp_info(exp_info)
    
    # Reward signal
    feedback_fn = lambda x, target: -((x[0] - target[0]) ** 2 + x[1] ** 2)

    state_size = 2
    dt0 = 0.1
    max_steps = 1000

    # Initialize environment
    if env_name == "HO":
        env = HarmonicOscillator(args.process_noise, args.obs_noise, n_obs)
    elif env_name == "DO":
        env = DuffingOscillator(args.process_noise, args.obs_noise, n_obs)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    obs_type = "full_obs" if n_obs == 2 else "partial_obs"

    # Define memory and readout variables
    memory_variables = [f"y{i}" for i in range(env.n_obs)] + [f"a{i}" for i in range(state_size)] + ["rpe"]
    readout_variables = [f"a{i}" for i in range(state_size)] + ["rpe"]

    # Define variable list and model
    if program == "GP-ODE":
        layer_sizes = jnp.array([state_size, env.n_control_inputs])
        variable_list = [memory_variables, readout_variables]
        fitness_function = ODEEvaluator(env, state_size, dt0, feedback_fn, solver=diffrax.GeneralShARK(), max_steps=max_steps)
    elif program == "GP-SDE":
        layer_sizes = jnp.array([state_size, env.n_control_inputs, state_size])
        variable_list = [memory_variables, readout_variables, memory_variables]
        fitness_function = SDEEvaluator(env, state_size, dt0, feedback_fn, gamma=args.gamma, solver=diffrax.GeneralShARK(), max_steps=max_steps)
    else:
        raise ValueError(f"Unknown program: {program}")

    # Define operator list (name, function, arity, probability)
    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
        ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1),
    ]

    # Initialize GP framework
    strategy = GeneticProgramming(
        num_generations=args.num_generations,
        population_size=args.population_size,
        fitness_function=fitness_function,
        operator_list=operator_list,
        variable_list=variable_list,
        layer_sizes=layer_sizes,
        num_populations=args.num_populations,
        device_type=jax.devices()[0].platform
    )

    for seed in range(10):
        print(f"\n{'=' * 60}")
        print(f"Program: {program} | Environment: {env_name} | Observation type: {obs_type} | Experiment id: {args.exp_id} | Seed: {seed}")
        print(f"{'=' * 60}")

        # Set random seed
        key = jr.PRNGKey(seed)
        key, init_key, data_key = jr.split(key, 3)

        # Generate data
        data = get_data(data_key, env, args.batch_size, args.dt, args.T, args.lambd, args.jump_size)

        strategy.reset()

        save_path = f"results/{env_name}/{obs_type}/Experiment_{args.exp_id}/{program}/seed_{seed}"
        os.makedirs(save_path, exist_ok=True)

        # Fit the model
        best_fitnesses = strategy.fit(init_key, data, verbose=True, save_pareto_front=True, path_to_file=save_path)

        best_idx = jnp.argmin(jnp.array(strategy.pareto_front[0]))
        jnp.save(f"{save_path}/best_candidate.npy", strategy.pareto_front[1][best_idx])
        jnp.save(f"{save_path}/best_fitnesses.npy", jnp.array(best_fitnesses))


if __name__ == "__main__":
    # Experiment H1: Fixed target, full observability
    run(program="GP-ODE", env_name="HO", exp_info="SDE-GP/config/HO/exp_01.txt", n_obs=2)
    run(program="GP-SDE", env_name="HO", exp_info="SDE-GP/config/HO/exp_01.txt", n_obs=2)

    # Experiment H2: Changing target, full observability
    run(program="GP-ODE", env_name="HO", exp_info="SDE-GP/config/HO/exp_02.txt", n_obs=2)
    run(program="GP-SDE", env_name="HO", exp_info="SDE-GP/config/HO/exp_02.txt", n_obs=2)

    # Experiment H3: fixed target, partial observability
    run(program="GP-ODE", env_name="HO", exp_info="SDE-GP/config/HO/exp_03.txt", n_obs=1)
    run(program="GP-SDE", env_name="HO", exp_info="SDE-GP/config/HO/exp_03.txt", n_obs=1)

    # Experiment H4: changing target, partial observability
    run(program="GP-ODE", env_name="HO", exp_info="SDE-GP/config/HO/exp_04.txt", n_obs=1)
    run(program="GP-SDE", env_name="HO", exp_info="SDE-GP/config/HO/exp_04.txt", n_obs=1)

    # Experiment D1: Fixed target, full observability
    run(program="GP-ODE", env_name="DO", exp_info="SDE-GP/config/DO/exp_01.txt", n_obs=2)
    run(program="GP-SDE", env_name="DO", exp_info="SDE-GP/config/DO/exp_01.txt", n_obs=2)

    # Experiment D2: Changing target, full observability
    run(program="GP-ODE", env_name="DO", exp_info="SDE-GP/config/DO/exp_02.txt", n_obs=2)
    run(program="GP-SDE", env_name="DO", exp_info="SDE-GP/config/DO/exp_02.txt", n_obs=2)

    # Experiment D3: fixed target, partial observability
    run(program="GP-ODE", env_name="DO", exp_info="SDE-GP/config/DO/exp_03.txt", n_obs=1)
    run(program="GP-SDE", env_name="DO", exp_info="SDE-GP/config/DO/exp_03.txt", n_obs=1)
    
    # Experiment D4: changing target, partial observability
    run(program="GP-ODE", env_name="DO", exp_info="SDE-GP/config/DO/exp_04.txt", n_obs=1)
    run(program="GP-SDE", env_name="DO", exp_info="SDE-GP/config/DO/exp_04.txt", n_obs=1)