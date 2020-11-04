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
from metaheuristic_environment import MetaheuristicEnvironment
from datetime import datetime
import numpy as npi
import errno
import shutil
import sys

class TrainableWrapper(Trainable):

    def _setup(self, config):
        self.config = config
        self.model = config['env_config']['trainable_class'](env=config['env'], config=config)
        self.metaheuristic_class = config['env_config']['metaheuristic_class']
        self.validation_data_folder = config['env_config']['validation_data_folder']
        if os.path.isdir(self.validation_data_folder) is False:
            Path(self.validation_data_folder).mkdir(parents=True)
        self.highest_mean_reward = -sys.float_info.max
        self.latest_checkpoint = None
        self.worker_id = str(uuid.uuid4())
        self.validation_test_results_filename = self.validation_data_folder + '/' + str(self.metaheuristic_class) + '_' + str(config['env_config']['problem'].training_instances[0]) + '_' + self.worker_id + '.txt'
        self.validation_test_results = open(self.validation_test_results_filename, 'w')
        self.validation_test_results.close()

    def compute_action(self,
                     observation,
                     state=None,
                     prev_action=None,
                     prev_reward=None,
                     info=None,
                     policy_id=None,
                     full_fetch=False):
        if policy_id is None:
            policy_id="default_policy"
        if state is None:
            state = []
        preprocessed = self.model.workers.local_worker().preprocessors[policy_id].transform(observation)
        filtered_obs = self.model.workers.local_worker().filters[policy_id](preprocessed, update=False)

        if state:
            return self.model.get_policy(policy_id).compute_single_action(
              filtered_obs,
              state,
              prev_action,
              prev_reward,
              info,
              clip_actions=self.model.config["clip_actions"])
        res = self.model.get_policy(policy_id).compute_single_action(
          filtered_obs,
          state,
          prev_action,
          prev_reward,
          info,
          clip_actions=self.model.config["clip_actions"])
        if full_fetch:
            return res
        else:
            return res[0]  # backwards compatibility

    def _train(self):
        print('CPU usage: ', psutil.cpu_percent(), '%')
        #print('Allocated threads: ', os.system("ps -eo nlwp | tail -n +2 | awk '{ num_threads += $1 } END { print num_threads }'"))
        result = self.model.train()
        workers = self.model.workers
        local_worker = workers.local_worker()
        test_instances = local_worker.env.problem.test_instances
        for test_func in test_instances:
            local_worker.env.problem.running_instances = [test_func]
            local_worker.env.problem.indexes = []
            local_worker.env.problem.next_instance()
            reward_sums_avg = 0
            best_fitnesses_ever_avg = 0
            best_fitnesses = []
            for i in range(21):
                state = local_worker.get_policy().get_initial_state()
                prev_action = np.zeros_like(local_worker.env.action_space.sample())
                prev_reward = 0
                info = {}
                obs = local_worker.env.reset()
                terminal = False
                while terminal is False:
                    action, state, fetch = self.compute_action(obs, state=state, prev_action=prev_action,
                                                                           prev_reward=prev_reward, info=info, full_fetch=True)
                    obs, reward, terminal, info = local_worker.env.step(action)
                    prev_action = action
                    prev_reward = reward
                if i > 0:
                    reward_sums_avg += info['reward_sum']
                    best_fitnesses_ever_avg += info['best_fitness_ever']
                    best_fitnesses.append(str(info['best_fitness_ever']))
            with open(self.validation_test_results_filename, 'a') as f:
                f.write(str(datetime.now()) + '\n')
                f.write('VALIDATION func' + str(test_func) + ': ' + str(result['episode_reward_mean']) + ', ' + str(reward_sums_avg / 20) + '\n')
                f.write(','.join(best_fitnesses) + '\n')
        return result

    def _save(self, checkpoint_dir):
        if os.path.isdir(checkpoint_dir) is False:
            Path(checkpoint_dir).mkdir(parents=True)
        print(f'Saving checkpoint at {checkpoint_dir}.')
        saved_model_path = self.model.save(checkpoint_dir)
        print(f'Checkpoint saved at {saved_model_path}.')
        return saved_model_path

    def _restore(self, checkpoint_path):
        print(f'Restoring checkpoint from {checkpoint_path}.')
        self.model.restore(checkpoint_path)
        print(f'Checkpoint restored from {checkpoint_path}.')
