import gym
import sys
import math
import json
import time
import uuid
import bisect
import random
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime
from gym.spaces import discrete, box

class MetaheuristicEnvironment(gym.Env):
    
    def __init__(self, env_config):
        parameters = {}
        self.logger = logging.getLogger(__name__)

        metaheuristic_config_file = env_config['metaheuristic_config_file']
        with open(metaheuristic_config_file, 'r') as f:
            metaheuristic_config_data = json.load(f)
        self.parameters_ranges = metaheuristic_config_data['parameters_ranges']
        self.constant_parameters = metaheuristic_config_data['constant_parameters']

        self.metaheuristic_class = env_config['metaheuristic_class']
        self.problem = env_config['problem']
        self.actions_between_0_1 = env_config['actions_between_0_1']
        self.reward_scaling_factor = env_config['reward_scaling_factor']
        min_ranges = []
        max_ranges = []
        for parameter in self.parameters_ranges:
            par_range = self.parameters_ranges[parameter]
            min_ranges.append(par_range[0])
            max_ranges.append(par_range[1])
            parameters[parameter] = random.random() * (par_range[1] - par_range[0]) + par_range[0]
        for parameter in self.constant_parameters:
            parameters[parameter] = self.constant_parameters[parameter]
        self.metaheuristic = self.metaheuristic_class(parameters, self.problem)
        self.previous_fitnesses = self.metaheuristic.get_fitnesses()
        #self.all_fitnesses_ever = np.sort(self.previous_fitnesses).tolist()
        current_min_fitness, current_max_fitness = self.metaheuristic.get_best_worst_fitnesses()
        self.last_max_fitness = current_max_fitness
        self.best_fitness_ever = current_max_fitness
        self.last_best_fitness_ever = self.best_fitness_ever
        self.best_position_ever = self.metaheuristic.get_best_position()
        self.iteration_counter = 0
        self.last_improvement_time = 0
        self.worst_fitness_ever = current_min_fitness
        self.last_worst_fitness_ever = self.worst_fitness_ever       
 
        self.init_budget = env_config['init_budget']
        self.current_budget = self.init_budget
        self.previous_fitness_improv_rate_length = 10
        self.previous_bsf_improvements_length = 10
        self.previous_fitness_improv_rate = np.zeros(self.previous_fitness_improv_rate_length)
        self.previous_bsf_improvements = np.zeros(self.previous_bsf_improvements_length)
        self.update_observables()
        self.observation_space = box.Box(low=np.full(len(self.observables), -float('inf')),
                                         high=np.full(len(self.observables), float('inf')),
                                         dtype=np.float32)
        if self.actions_between_0_1:
            self.action_space = box.Box(low=np.zeros(len(min_ranges)), high=np.full(shape=(len(min_ranges),), fill_value=1), dtype=np.float32)
        else:
            self.action_space = box.Box(low=np.array(min_ranges), high=np.array(max_ranges), dtype=np.float32)
        self.sum_reward = 0

    def generate_random_parameters(self):
        gen_parameters = []
        for parameter in self.parameters_ranges:
            range_ = self.parameters_ranges[parameter]
            gen_parameters.append(random.random() * (range_[1] - range_[0]) + range_[0])
        return np.array(gen_parameters)
    
    def update_observables(self):
        self.update_best_and_worst_fitnesses()
        self.observables = []
        #self.observables = [self.worst_fitness_ever, self.best_fitness_ever]
        #for i in range(11):
        #    self.observables.append(np.percentile(self.all_fitnesses_ever, i * 10))         
        self.observables += [self.avg_fitness()]
        self.observables += [self.fitness_std()]
        self.observables += [self.remaining_budget()]
        self.observables += [self.stagnation()]
        self.observables += self.random_distances()
        self.observables += self.best_distance()
        self.observables += self.random_fitness_distances()
        self.observables += self.random_fitness_distances_best()
        self.observables += self.best_distance_bsf()
        self.observables += self.fitness_improvements_ratio()
        self.observables += self.bsf_improvements()
        
    def update_best_and_worst_fitnesses(self):
        current_min_fitness, current_max_fitness = self.metaheuristic.get_best_worst_fitnesses()
        self.last_best_fitness_ever = self.best_fitness_ever
        self.last_worst_fitness_ever = self.worst_fitness_ever
        if current_max_fitness > self.best_fitness_ever:
            self.best_fitness_ever = current_max_fitness
            self.best_position_ever = self.metaheuristic.get_best_position()
        if current_min_fitness < self.worst_fitness_ever:
            self.worst_fitness_ever = current_min_fitness
        if self.best_fitness_ever == self.worst_fitness_ever:
            self.best_fitness_ever += 0.000000001
        
    def reset(self):
        del self.metaheuristic
        self.problem.next_instance()
        self.sum_reward = 0
        parameters = {}
        for parameter in self.parameters_ranges:
            par_range = self.parameters_ranges[parameter]
            parameters[parameter] = random.random() * (par_range[1] - par_range[0]) + par_range[0]
        for parameter in self.constant_parameters:
            parameters[parameter] = self.constant_parameters[parameter]
        self.metaheuristic = self.metaheuristic_class(parameters, self.problem)
        self.previous_fitnesses = self.metaheuristic.get_fitnesses()
        #self.all_fitnesses_ever = np.sort(self.previous_fitnesses).tolist()
        current_min_fitness, current_max_fitness = self.metaheuristic.get_best_worst_fitnesses()
        self.last_max_fitness = current_max_fitness
        self.best_fitness_ever = current_max_fitness
        self.last_best_fitness_ever = self.best_fitness_ever
        self.best_position_ever = self.metaheuristic.get_best_position()
        self.iteration_counter = 0
        self.last_improvement_time = 0
        self.worst_fitness_ever = current_min_fitness
        self.last_worst_fitness_ever = self.worst_fitness_ever
        
        self.beginning_time = time.time()
        self.current_budget = self.init_budget
        self.previous_fitness_improv_rate = np.zeros(self.previous_fitness_improv_rate_length)
        self.previous_bsf_improvements = np.zeros(self.previous_bsf_improvements_length)
        self.update_observables()
        self.sum_reward = 0        
        return self.observables
        
    def step(self, action):
        new_parameters = {}
        par_index = 0
        for parameter in self.parameters_ranges:
            if self.actions_between_0_1:
                new_parameters[parameter] = action[par_index] * (self.parameters_ranges[parameter][1] - self.parameters_ranges[parameter][0]) + self.parameters_ranges[parameter][0]
            else:
                new_parameters[parameter] = action[par_index]
            par_index += 1
        for parameter in self.constant_parameters:
            new_parameters[parameter] = self.constant_parameters[parameter]
        
        self.metaheuristic.set_parameters(new_parameters)
        
        self.previous_fitnesses = self.metaheuristic.get_fitnesses()
        n_iterations = self.metaheuristic.run(self.current_budget)
        self.current_budget -= n_iterations
        self.update_observables()
        
        current_min_f, current_max_f = self.metaheuristic.get_best_worst_fitnesses()
        if current_max_f > 0 and self.last_max_fitness > 0:
            current_max_fitness = 1.0 / max([current_max_f, 10e-20])
            previous_max_fitness = 1.0 / max([self.last_max_fitness, 10e-20])
        else:
            current_max_fitness = 1.0 / max([-current_max_f, 10e-20])
            previous_max_fitness = 1.0 / max([-self.last_max_fitness, 10e-20])
        ret = self.observables, self.reward_scaling_factor * math.log10(current_max_fitness / previous_max_fitness), self.current_budget <= 0, {'best_fitness_ever': self.best_fitness_ever}
        #if ret[1] == 0:
        #    print(ret[1], current_max_fitness, previous_max_fitness, -current_max_f, -self.last_max_fitness, current_max_fitness / previous_max_fitness, math.log10(current_max_fitness / previous_max_fitness))        
        self.last_max_fitness = current_max_f
        self.sum_reward += ret[1]
        ret[3]['reward_sum'] = self.sum_reward
        self.iteration_counter += 1

        return ret

    # Feature #0
    def avg_fitness(self):
        fitnesses = self.metaheuristic.get_fitnesses()
        fitness_avg = np.mean(fitnesses)
        return (self.best_fitness_ever - fitness_avg) / (self.best_fitness_ever - self.worst_fitness_ever)

    # Feature #1
    def fitness_std(self):
        fitnesses = self.metaheuristic.get_fitnesses()
        fitness_std = np.std(fitnesses)
        max_std = np.std([self.worst_fitness_ever, self.best_fitness_ever])
        return fitness_std / max_std

    # Feature #2
    def remaining_budget(self):
        return self.current_budget / self.init_budget

    # Feature #3
    def stagnation(self):
        return (self.iteration_counter - self.last_improvement_time) / self.init_budget

    # Features #4-8
    def random_distances(self):
        population = self.metaheuristic.get_population()
        pop_size = population.shape[0]
        dimensions = population.shape[1]
        max_distance = self.problem.max_distance()
        distances = []
        for i in range(100):
            random_i = random_j = 0
            while random_i == random_j:
                random_i = random.randint(0, pop_size - 1)
                random_j = random.randint(0, pop_size - 1)
            vector_i = population[random_i]
            vector_j = population[random_j]
            distances.append(self.problem.distance(vector_i, vector_j) / max_distance)
        return distances

    #Features #9-13
    def best_distance(self):
        population = self.metaheuristic.get_population()
        fitnesses = self.metaheuristic.get_fitnesses()
        pop_size = population.shape[0]
        dimensions = population.shape[1]
        best_indiv_index = np.argmax(fitnesses)
        best_indiv = population[best_indiv_index]
        max_distance = self.problem.max_distance()
        distances = []
        for i in range(10):
            random_i = best_indiv_index
            while random_i == best_indiv_index:
                random_i = random.randint(0, pop_size - 1)
            vector_i = population[random_i]
            distances.append(self.problem.distance(vector_i, best_indiv) / max_distance)
        return distances

    #Features #14-18
    def random_fitness_distances(self):
        fitnesses = self.metaheuristic.get_fitnesses()
        pop_size = fitnesses.shape[0]
        diffs = []
        for i in range(100):
            random_i = random_j = 0
            while random_i == random_j:
                random_i = random.randint(0, pop_size - 1)
                random_j = random.randint(0, pop_size - 1)
            diffs.append(abs(fitnesses[random_i]-fitnesses[random_j]) / (self.best_fitness_ever - self.worst_fitness_ever))
        return diffs

    #Features #19-23
    def random_fitness_distances_best(self):
        fitnesses = self.metaheuristic.get_fitnesses()
        pop_size = fitnesses.shape[0]
        best_indiv_index = np.argmax(fitnesses)
        diffs = []
        for i in range(10):
            random_i = best_indiv_index
            while random_i == best_indiv_index:
                random_i = random.randint(0, pop_size - 1)
            diffs.append(abs(fitnesses[random_i]-fitnesses[best_indiv_index]) / (self.best_fitness_ever - self.worst_fitness_ever))
        return diffs

    #Features #24-28
    def best_distance_bsf(self):
        population = self.metaheuristic.get_population()
        pop_size = population.shape[0]
        dimensions = population.shape[1]
        max_distance = self.problem.max_distance()
        distances = []
        for i in range(10):
            random_i = random.randint(0, pop_size - 1)
            vector_i = population[random_i]
            distances.append(self.problem.distance(vector_i, self.best_position_ever) / max_distance)
        return distances

    #Features #29-38
    def fitness_improvements_ratio(self):
        fitnesses = self.metaheuristic.get_fitnesses()
        ratio = np.sum(np.array(fitnesses) > np.array(self.previous_fitnesses)) / len(fitnesses)
        self.previous_fitness_improv_rate = np.delete(self.previous_fitness_improv_rate,
                                                      self.previous_fitness_improv_rate.shape[0] - 1)
        self.previous_fitness_improv_rate = np.insert(self.previous_fitness_improv_rate, 0, ratio)
        return self.previous_fitness_improv_rate.tolist()

    #Features #39-47
    def bsf_improvements(self):
        new_value = 0
        if self.last_best_fitness_ever < self.best_fitness_ever:
            new_value = 1
        self.previous_bsf_improvements = np.delete(self.previous_bsf_improvements,
                                                   self.previous_bsf_improvements.shape[0] - 1)
        self.previous_bsf_improvements = np.insert(self.previous_bsf_improvements, 0, new_value)
        return self.previous_bsf_improvements.tolist()

