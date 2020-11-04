import uuid
import os
import pickle
import psutil
from pathlib import Path
from scipy.stats import loguniform
import numpy as np
from ray import tune
import random, sys, gym, ray
from ray.rllib.agents import dqn, ddpg
from ray.rllib.agents.ddpg import td3, apex
from ray.tune import run, sample_from, Trainable
from ray.tune.schedulers import PopulationBasedTraining
from metaheuristics import HCLPSO, FSS, BinGA, DE, ACO_TSP
from metaheuristic_environment import MetaheuristicEnvironment
from cec17 import CEC17
from tsp import TSP
from knapsack import Knapsack
from datetime import datetime
import numpy as npi
import errno
import shutil
import sys

print(sys.argv)
redis_address = sys.argv[1]
redis_password = '123456'
trainable_class = eval(sys.argv[2])
metaheuristic_class = eval(sys.argv[3])
metaheuristic_config_file = sys.argv[4]
problem_config_file = sys.argv[5]
problem_class = eval(sys.argv[6])
pbt_num_samples = int(sys.argv[7])
num_workers_per_sample = int(sys.argv[8])
pbt_run_name = sys.argv[9]
validation_data_folder = sys.argv[10]
reward_scaling_factor = float(sys.argv[11])

print('Starting experiment with the following setup: ' + str(sys.argv[1:]))

print(str(problem_class))
env_config = {"metaheuristic_class": metaheuristic_class,
              "metaheuristic_config_file": metaheuristic_config_file,
              "problem": problem_class(problem_config_file),
              "init_budget": 300,
              "trainable_class": trainable_class,
              "actions_between_0_1": True,
              "validation_data_folder": validation_data_folder,
              "reward_scaling_factor": reward_scaling_factor}

ray.init(address=redis_address, redis_password='123456', lru_evict=True)

pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=10,
        resample_probability=0.25,
        hyperparam_mutations={
            "critic_lr": lambda: loguniform.rvs(1e-6, 1e-1),
            "actor_lr": lambda: loguniform.rvs(1e-6, 1e-1),
            "tau": lambda: np.random.random() * (0.01 - 0.0001) + 0.0001,
            "l2_reg": lambda: np.random.random()
        })

config={
        "env": MetaheuristicEnvironment,
        "env_config": env_config,
        "target_network_update_freq": tune.grid_search([int(value) for value in np.arange(100, 5001, (5001-100) / pbt_num_samples)]),
        "timesteps_per_iteration": 45000,
        "learning_starts": 45000,
        "train_batch_size": 45000,
        "sample_batch_size": 300,
        "buffer_size": 45000,
        "num_workers": num_workers_per_sample,
        "num_gpus": 0,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "critic_lr": 1e-4,
        "actor_lr": 1e-4,
        "tau": 2e-4,
        "actor_hiddens": [300, 300, 300, 300],
        "critic_hiddens": [300, 300, 300, 300],
        "l2_reg": 1e-6,
        "local_tf_session_args": {
              "intra_op_parallelism_threads": 1,
              "inter_op_parallelism_threads": 1,
              }
        }

if trainable_class == td3.TD3Trainer:
    config['exploration_config'] = {'random_timesteps': 45000}
print('OMP_NUM_THREADS:', os.environ['OMP_NUM_THREADS'])
run(
    TrainableWrapper,
    name=pbt_run_name,
    scheduler=pbt,
    num_samples=1,
    config=config,
    resources_per_trial={"cpu": 1, "extra_cpu": num_workers_per_sample}
)
