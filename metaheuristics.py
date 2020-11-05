import tsp
import sys
import gym
import time
import math
import random
import numpy as np
from scipy.spatial import distance
from gym.spaces import discrete, box
from utils import modify_population_size

class HCLPSO:
    def __init__(self, parameters, problem):
        self.problem = problem
        self.w = parameters['w']
        self.c = parameters['c']
        self.c1 = parameters['c1']
        self.c2 = parameters['c2']
        self.m = int(round(parameters['m']))
        self.search_space_bounds_ = self.problem.search_space_bounds()[0]
        self.vel_min = -(self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4
        self.vel_max = (self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4
        self.pop_explr_size = int(round(parameters['pop_explr_size']))
        self.pop_explt_size = int(round(parameters['pop_explt_size']))
        self.v = parameters['v']
        self.previous_explr_size = self.pop_explr_size
        self.previous_explt_size = self.pop_explt_size
        self.pop_explr = np.random.random((self.pop_explr_size, self.problem.get_dimensions())) * ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4) + ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) * 3 / 4) + self.search_space_bounds_[0]
        self.pop_explt = np.random.random((self.pop_explt_size, self.problem.get_dimensions())) * ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4) + ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) * 3 / 4) + self.search_space_bounds_[0]
        self.vel_explr = np.zeros((self.pop_explr_size, self.problem.get_dimensions()))
        self.vel_explt = np.zeros((self.pop_explt_size, self.problem.get_dimensions()))
        self.fitnesses_explr = np.array([self.problem.evaluate(individual) for individual in self.pop_explr])
        self.fitnesses_explt = np.array([self.problem.evaluate(individual) for individual in self.pop_explt])
        self.pbests_explr = np.copy(self.pop_explr)
        self.pbests_explt = np.copy(self.pop_explt)
        self.effective_pbests_explr = np.copy(self.pbests_explr)
        self.effective_pbests_explt = np.copy(self.pbests_explt)
        self.pbests_fitnesses_explr = np.copy(self.fitnesses_explr)
        self.pbests_fitnesses_explt = np.copy(self.fitnesses_explt)
        self.no_improv_counters_explr = np.zeros((self.pop_explr_size,))
        self.no_improv_counters_explt = np.zeros((self.pop_explt_size,))
        self.gbest = np.zeros((self.problem.get_dimensions(),))
        self.gbest_fitness = -sys.float_info.max
        self.best_fitness_ever = -sys.float_info.max
        
    def __str__(self):
        return 'HCLPSO'

    def get_best_worst_fitnesses(self):
        concatenated_data = np.concatenate([self.fitnesses_explr, self.fitnesses_explt])
        return concatenated_data.min(), concatenated_data.max()
    
    def get_best_position(self):
        highest_fitness_explr = np.max(self.fitnesses_explr)
        highest_fitness_explt = np.max(self.fitnesses_explt)
        if highest_fitness_explr > highest_fitness_explt:
            highest_fitness_index = np.argmax(self.fitnesses_explr)
            return self.pop_explr[highest_fitness_index]
        else:
            highest_fitness_index = np.argmax(self.fitnesses_explt)
            return self.pop_explt[highest_fitness_index]
    
    def get_population(self):
        return np.concatenate([self.pop_explr, self.pop_explt])
    
    def get_fitnesses(self):
        return np.concatenate([self.fitnesses_explr, self.fitnesses_explt])
        
    def update_gbest(self):
        highest_fitness_explr = np.max(self.fitnesses_explr)
        highest_fitness_explt = np.max(self.fitnesses_explt)
        if highest_fitness_explr > self.gbest_fitness or highest_fitness_explt > self.gbest_fitness:
            if highest_fitness_explr > highest_fitness_explt:
                self.gbest_fitness = highest_fitness_explr
                highest_fitness_index = np.argmax(self.fitnesses_explr)
                self.gbest = np.copy(self.pop_explr[highest_fitness_index])
            else:
                self.gbest_fitness = highest_fitness_explt
                highest_fitness_index = np.argmax(self.fitnesses_explt)
                self.gbest = np.copy(self.pop_explt[highest_fitness_index])
            
    def update_pbests(self):
        for i in range(self.pbests_fitnesses_explr.shape[0]):
            if self.fitnesses_explr[i] > self.pbests_fitnesses_explr[i]:
                self.pbests_fitnesses_explr[i] = self.fitnesses_explr[i]
                self.pbests_explr[i] = np.copy(self.pop_explr[i])
                self.no_improv_counters_explr[i] = 0
            else:
                self.no_improv_counters_explr[i] += 1
        for i in range(self.pbests_fitnesses_explt.shape[0]):
            if self.fitnesses_explt[i] > self.pbests_fitnesses_explt[i]:
                self.pbests_fitnesses_explt[i] = self.fitnesses_explt[i]
                self.pbests_explt[i] = np.copy(self.pop_explt[i])
                self.no_improv_counters_explt[i] = 0
            else:
                self.no_improv_counters_explt[i] += 1
                
    def build_effective_pbest_vector(self, individual_index):
        following_probability = 0.25 * ((np.exp((10 * (individual_index - 1)) / (self.pop_explr.shape[0] + self.pop_explt.shape[0] - 1)) - 1) / (np.exp(10) - 1))
        learning_bool = np.random.random((self.problem.get_dimensions(),)) < following_probability
        individual_index -= 1
        if learning_bool.any() == False:
            learning_bool[np.random.choice(range(self.problem.get_dimensions()))] = True
        converted_index = individual_index - self.effective_pbests_explr.shape[0] if individual_index >= self.effective_pbests_explr.shape[0] else individual_index
        for i in range(learning_bool.shape[0]):
            if learning_bool[i]:
                chosen_indexes = np.random.choice(range(self.pbests_explr.shape[0]), 2)
                max_index = np.argmax(self.pbests_fitnesses_explr[chosen_indexes])
                if individual_index < self.effective_pbests_explr.shape[0]:
                    self.effective_pbests_explr[converted_index][i] = self.pbests_explr[chosen_indexes[max_index]][i]
                else:
                    self.effective_pbests_explt[converted_index][i] = self.pbests_explr[chosen_indexes[max_index]][i]
            else:
                if individual_index < self.effective_pbests_explr.shape[0]:
                    self.effective_pbests_explr[converted_index][i] = self.pbests_explr[converted_index][i]
                else:
                    self.effective_pbests_explt[converted_index][i] = self.pbests_explt[converted_index][i]
            
    def update_effective_pbests(self):
        for i in range(1, self.pop_explr.shape[0] + 1):
            if self.no_improv_counters_explr[i - 1] >= self.m:
                self.no_improv_counters_explr[i - 1] = 0
                self.build_effective_pbest_vector(i)
        for i in range(1, self.pop_explt.shape[0] + 1):
            if self.no_improv_counters_explt[i - 1] >= self.m:
                self.no_improv_counters_explt[i - 1] = 0
                self.build_effective_pbest_vector(self.pop_explr.shape[0] + i)
    
    def update_velocity(self):
        self.vel_explr = self.w * self.vel_explr + self.c * np.random.random((self.problem.get_dimensions(),)) * (self.effective_pbests_explr - self.pop_explr)
        replicated_gbest = np.repeat(np.array([self.gbest]), self.pop_explt.shape[0], axis=0)
        self.vel_explt = self.w * self.vel_explt + self.c1 * np.random.random((self.problem.get_dimensions(),)) * (self.effective_pbests_explt - self.pop_explt) + self.c2 * np.random.random((self.problem.get_dimensions(),)) * (replicated_gbest - self.pop_explt)
        vel_explr_lower_min = self.vel_explr < self.vel_min
        vel_explr_greater_max = self.vel_explr > self.vel_max
        vel_explt_lower_min = self.vel_explt < self.vel_min
        vel_explt_greater_max = self.vel_explt > self.vel_max
        self.vel_explr[vel_explr_greater_max] = self.vel_max
        self.vel_explr[vel_explr_lower_min] = self.vel_min
        self.vel_explt[vel_explt_greater_max] = self.vel_max
        self.vel_explt[vel_explt_lower_min] = self.vel_min
        
    def update_position(self):
        self.pop_explr += self.vel_explr
        self.pop_explt += self.vel_explt
        pop_explr_lower_min = self.pop_explr < self.search_space_bounds_[0]
        pop_explr_greater_max = self.pop_explr > self.search_space_bounds_[1]
        pop_explt_lower_min = self.pop_explt < self.search_space_bounds_[0]
        pop_explt_greater_max = self.pop_explt > self.search_space_bounds_[1]
        self.vel_explr[pop_explr_greater_max | pop_explr_lower_min] *= -1
        self.vel_explt[pop_explt_greater_max | pop_explt_lower_min] *= -1
        self.pop_explr[pop_explr_greater_max] = self.search_space_bounds_[1]
        self.pop_explr[pop_explr_lower_min] = self.search_space_bounds_[0]
        self.pop_explt[pop_explt_greater_max] = self.search_space_bounds_[1]
        self.pop_explt[pop_explt_lower_min] = self.search_space_bounds_[0]
        
    def update_fitnesses(self):
        self.fitnesses_explr = np.array([self.problem.evaluate(individual) for individual in self.pop_explr])
        self.fitnesses_explt = np.array([self.problem.evaluate(individual) for individual in self.pop_explt])
    
    def modify_population(self):
        pop_matrices_data = [self.vel_explr, self.pbests_explr, self.effective_pbests_explr,
                             self.pbests_fitnesses_explr, self.no_improv_counters_explr]
        # False: does not add or remove column
        pop_matrices_remove_add_column = [False, False, False, False, False]
        # False: does not copy data (set all values to zero); True: copies data
        pop_matrices_gen_copy = [False, True, True, True, False]
        self.pop_explr, self.fitnesses_explr, pop_matrices_data = modify_population_size(self.pop_explr,self.fitnesses_explr,
                               lambda x: self.problem.evaluate(x),
                               self.pop_explr_size - self.previous_explr_size, self.v,
                               np.repeat([self.search_space_bounds_], self.problem.get_dimensions(), 0),
                               pop_matrices_data, pop_matrices_remove_add_column, pop_matrices_gen_copy)
        self.vel_explr = pop_matrices_data[0]
        self.pbests_explr = pop_matrices_data[1]
        self.effective_pbests_explr = pop_matrices_data[2]
        self.pbests_fitnesses_explr = pop_matrices_data[3]
        self.no_improv_counters_explr = pop_matrices_data[4]        
        pop_matrices_data = [self.vel_explt, self.pbests_explt, self.effective_pbests_explt,
                             self.pbests_fitnesses_explt, self.no_improv_counters_explt]
        # False: does not add or remove column
        pop_matrices_remove_add_column = [False, False, False, False, False]
        # False: does not copy data (set all values to zero); True: copies data
        pop_matrices_gen_copy = [False, True, True, True, False]
        self.pop_explt, self.fitnesses_explt, pop_matrices_data = modify_population_size(self.pop_explt, self.fitnesses_explt,
                               lambda x: self.problem.evaluate(x),
                               self.pop_explt_size - self.previous_explt_size, self.v,
                               np.repeat([self.search_space_bounds_], self.problem.get_dimensions(), 0),
                               pop_matrices_data, pop_matrices_remove_add_column, pop_matrices_gen_copy)
        self.vel_explt = pop_matrices_data[0]
        self.pbests_explt = pop_matrices_data[1]
        self.effective_pbests_explt = pop_matrices_data[2]
        self.pbests_fitnesses_explt = pop_matrices_data[3]
        self.no_improv_counters_explt = pop_matrices_data[4]
        
    def set_parameters(self, parameters):        
        self.w = parameters['w']
        self.c = parameters['c']
        self.c1 = parameters['c1']
        self.c2 = parameters['c2']
        self.m = int(round(parameters['m']))
        self.vel_min = -(self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4
        self.vel_max = (self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4
        self.previous_explr_size = self.pop_explr_size
        self.previous_explt_size = self.pop_explt_size
        self.pop_explr_size = int(round(parameters['pop_explr_size'], 0))
        self.pop_explt_size = int(round(parameters['pop_explt_size'], 0))
        self.v = parameters['v']
 
    def run(self, current_budget):
        self.update_gbest()
        self.update_pbests()
        self.update_effective_pbests()
        self.update_velocity()
        self.update_position()
        self.update_fitnesses()
        self.best_fitness_ever = self.gbest_fitness
        return 1

class FSS:
    def __init__(self, parameters, problem):
        self.problem = problem
        self.search_space_bounds_ = self.problem.search_space_bounds()[0]
        self.indiv_step = parameters['indiv_step'] * (self.search_space_bounds_[1] - self.search_space_bounds_[0])
        self.vol_step = parameters['vol_step'] * (self.search_space_bounds_[1] - self.search_space_bounds_[0])
        self.max_weight = parameters['max_weight']    
        self.pop_size = int(round(parameters['pop_size']))
        # Not used modify_population parameter (which is not implemented)
        self.v = parameters['v']
        self.population = np.random.random((self.pop_size, self.problem.get_dimensions())) * \
                ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4) + \
                ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) * 3 / 4) + \
                self.search_space_bounds_[0]
        self.weights = np.ones((self.pop_size,))
        self.weight_sum = np.sum(self.weights)
        self.fitnesses = np.array([self.problem.evaluate(individual) for individual in self.population])
        self.best_fitness_ever = self.fitnesses.max()

    def __str__(self):
        return 'FSS'

    def get_best_worst_fitnesses(self):
        return self.fitnesses.min(), self.fitnesses.max()
    
    def get_best_position(self):
        highest_fitness_index = np.argmax(self.fitnesses)
        return self.population[highest_fitness_index]
    
    def get_population(self):
        return self.population
    
    def get_fitnesses(self):
        return self.fitnesses
    
    def modify_population(self):
        raise NotImplementedError

    def set_parameters(self, parameters):
        self.indiv_step = parameters['indiv_step'] * (self.search_space_bounds_[1] - self.search_space_bounds_[0])
        self.vol_step = parameters['vol_step'] * (self.search_space_bounds_[1] - self.search_space_bounds_[0])
        self.max_weight = parameters['max_weight']
        # Population size must not change, unless modify_population is implemented
        self.pop_size = int(round(parameters['pop_size']))
        # Not used modify_population parameter (which is not implemented)
        self.v = parameters['v']


    def calculate_fitnesses(self, population):
        fitnesses = np.array([self.problem.evaluate(individual) for individual in population])
        return fitnesses

    def run(self, current_budget):
        #print(self.indiv_step, self.vol_step)
        fitnesses_before_indiv = self.calculate_fitnesses(self.population)
        new_population = self.population + ((np.random.random((self.pop_size,
                                                               self.problem.get_dimensions())) * 2) - 1) * self.indiv_step
        fitnesses_after_indiv = self.calculate_fitnesses(new_population)
        delta_f = fitnesses_after_indiv - fitnesses_before_indiv
        positive_delta_f = (delta_f > 0).astype(int)
        delta_f *= positive_delta_f
        tiled_positive_delta_f = np.tile(positive_delta_f.reshape((-1,1)), [1,self.problem.get_dimensions()])
        displacement = new_population - self.population
        displacement *= tiled_positive_delta_f
        self.population += displacement
        positive_delta_f_inf = np.where(positive_delta_f == 0, np.Inf, positive_delta_f)
        self.fitnesses = np.maximum(fitnesses_after_indiv * positive_delta_f_inf, self.fitnesses)
        max_delta_f = delta_f.max()
        self.weights += np.minimum(positive_delta_f * (delta_f / (max_delta_f + 0.00000001)), 5000)
        new_weight_sum = np.sum(self.weights)
        delta_f_sum = np.sum(delta_f) + 0.00000001
        tiled_delta_f = np.tile(delta_f.reshape(-1,1), [1,self.problem.get_dimensions()])
        i_vector = np.sum(tiled_delta_f * displacement, axis=0) / delta_f_sum
        tiled_i_vector = np.tile(i_vector.reshape((1,-1)), [self.pop_size,1])
        self.population += tiled_i_vector
        tiled_weights = np.tile(self.weights.reshape(-1,1), [1,self.problem.get_dimensions()])
        barycenter = np.sum(tiled_weights * self.population, axis=0) / new_weight_sum
        distances = np.array(list(map(lambda x: np.linalg.norm(x - barycenter), self.population)))
        tiled_distances = np.tile(distances.reshape((-1,1)), [1,self.problem.get_dimensions()])
        tiled_barycenter = np.tile(barycenter.reshape((1,-1)), [self.pop_size,1])
        vol_displacement = np.random.random((self.pop_size, self.problem.get_dimensions())) * \
                self.vol_step * ((self.population - tiled_barycenter) / tiled_distances)
        #self.population = self.population - vol_displacement if new_weight_sum > self.weight_sum else self.population + vol_displacement
        self.population += vol_displacement
        self.weight_sum = new_weight_sum
        self.population = np.maximum(self.population, self.search_space_bounds_[0])
        self.population = np.minimum(self.population, self.search_space_bounds_[1])
        max_fitness = self.fitnesses.max()
        if max_fitness > self.best_fitness_ever:
            self.best_fitness_ever = max_fitness
        return 1



class BinGA:

    def __init__(self, parameters, problem):
        self.problem = problem
        self.pop_size = int(parameters['pop_size'])
        self.prob_mutation_att = parameters['prob_mutation_att']
        self.crossover_prob = parameters['crossover_prob']
        self.elitism_size = int(round(parameters['elitism_size']))
        self.v = parameters['v']
        self.dimensions = self.problem.get_dimensions()

        self.population = np.random.randint(0, 2, (self.pop_size, self.dimensions))
        self.fitnesses = np.array([self.problem.evaluate(individual) for individual in self.population])
        self.best_fitness_ever = self.fitnesses.max()
        self.best_position_ever = self.get_best_position()

    def __str__(self):
        return 'BinGA'

    def get_best_worst_fitnesses(self):
        return self.fitnesses.min(), self.fitnesses.max()
    
    def get_best_position(self):
        highest_fitness_index = np.argmax(self.fitnesses)
        return self.population[highest_fitness_index]
    
    def get_population(self):
        return self.population
    
    def get_fitnesses(self):
        return self.fitnesses
    
    def modify_population(self):
        raise NotImplementedError

    def set_parameters(self, parameters):
        self.pop_size = int(parameters['pop_size'])
        self.prob_mutation_att = parameters['prob_mutation_att']
        self.crossover_prob = parameters['crossover_prob']
        self.elitism_size = int(round(parameters['elitism_size']))
        # Not used modify_population parameter (which is not implemented)
        self.v = parameters['v']

    def roulette_selection(self, k, elitism):
        selected = []
        indexes = list(range(self.pop_size))
        if elitism != 0:
            sorted_indexes = sorted(zip(indexes, self.fitnesses), key=lambda x: x[1], reverse=True)
            for i in range(elitism):
                selected.append(self.population[sorted_indexes[i][0]])
                indexes.remove(sorted_indexes[i][0])
        for i in range(k):
            if len(indexes) == 1:
                selected.append(self.population[indexes[0]])
            else:
                fitness_sum = self.fitnesses[indexes].sum()
                normalized_fitnesses_sum = 0
                random_value = np.random.random()
                for j, index in enumerate(indexes):
                    normalized_fitnesses_sum += self.fitnesses[index] / fitness_sum
                    if random_value <= normalized_fitnesses_sum:
                        selected.append(self.population[index])
                        del indexes[j]
                        break
        return selected

    def one_point_crossover(self, a, b):
        point = np.random.randint(1, self.dimensions - 1)
        return np.concatenate([a[:point], b[point:]]), np.concatenate([b[:point], a[point:]])

    def mutation(self, individual):
        for i in range(self.dimensions):
            if np.random.random() < self.prob_mutation_att:
                individual[i] = int(not(individual[i]))
        return individual

    def run(self, current_budget):
        offspring = []
        for i in range(self.pop_size):
            if np.random.random() < self.crossover_prob:
                selected = self.roulette_selection(2, 0)
                child_1, child_2 = self.one_point_crossover(selected[0], selected[1])
                offspring += [child_1, child_2]
        if len(offspring) != 0:
            offspring = np.array(offspring)
            self.population = np.concatenate([self.population, offspring])
        for i in range(len(self.population)):
            self.population[i] = self.mutation(self.population[i])
        self.population = np.array(self.roulette_selection(self.pop_size - self.elitism_size, self.elitism_size))
        self.fitnesses = np.array([self.problem.evaluate(individual) for individual in self.population])
        max_fitness = self.fitnesses.max()

        if max_fitness > self.best_fitness_ever:
            self.best_fitness_ever = max_fitness
            self.best_position_ever = self.get_best_position()
        return 1


class DE:

    def __init__(self, parameters, problem):
        self.problem = problem
        self.pop_size = int(parameters['pop_size'])
        self.f = parameters['f']
        self.cr = parameters['cr']
        self.v = parameters['v']
        self.dimensions = self.problem.get_dimensions()
        self.search_space_bounds_ = self.problem.search_space_bounds()[0]

        self.population = np.random.random((self.pop_size, self.dimensions)) * \
                ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) / 4) + \
                ((self.search_space_bounds_[1] - self.search_space_bounds_[0]) * 3 / 4) + \
                self.search_space_bounds_[0]
        self.fitnesses = np.array([self.problem.evaluate(individual) for individual in self.population])
        self.best_fitness_ever = self.fitnesses.max()
        
    def __str__(self):
        return 'DE'

    def get_best_worst_fitnesses(self):
        return self.fitnesses.min(), self.fitnesses.max()
    
    def get_best_position(self):
        highest_fitness_index = np.argmax(self.fitnesses)
        return self.population[highest_fitness_index]
    
    def get_population(self):
        return self.population
    
    def get_fitnesses(self):
        return self.fitnesses
    
    def modify_population(self):
        raise NotImplementedError

    def set_parameters(self, parameters):
        self.pop_size = int(parameters['pop_size'])
        self.f = parameters['f']
        self.cr = parameters['cr']
        # Not used modify_population parameter (which is not implemented)
        self.v = parameters['v']

    def run(self, current_budget):
        highest_fitness_index = np.argmax(self.fitnesses)
        best_position = self.population[highest_fitness_index]
        for i in range(self.pop_size):
            indexes_fitnesses = list(range(len(self.fitnesses)))
            indexes_fitnesses.remove(highest_fitness_index)
            selected_indexes = random.sample(indexes_fitnesses, 2)
            selected_positions = self.population[selected_indexes]
            mutant_vector = best_position + self.f * (selected_positions[0] - selected_positions[1])
            jrand = np.random.randint(0, self.dimensions)
            offspring = [(mutant_vector[j] if np.random.random() <= self.cr or j == jrand else self.population[i][j]) for j in range(self.dimensions)]
            offspring_fitness = self.problem.evaluate(offspring)
            if offspring_fitness > self.fitnesses[i]:
                self.population[i] = offspring
                self.fitnesses[i] = offspring_fitness
        max_fitness = self.fitnesses.max()
        if max_fitness > self.best_fitness_ever:
            self.best_fitness_ever = max_fitness
        return 1


class ACO_TSP:

    def __init__(self, parameters, problem):
        self.problem = problem
        self.pop_size = int(parameters['pop_size'])
        self.alpha = parameters['alpha']
        self.beta = parameters['beta']
        self.rho = parameters['rho']
        self.update_with_best_ever_prob = parameters['update_with_best_ever_prob']
        self.v = parameters['v']

        self.dimensions = self.problem.get_dimensions()

        self.pheromone_matrix = np.zeros((self.dimensions, self.dimensions))
        self.population = np.array([random.sample(range(self.dimensions),
                                                  self.dimensions) for i in range(self.pop_size)])
        self.fitnesses = np.array([self.problem.evaluate(individual) for individual in self.population])
        self.best_fitness_ever = self.fitnesses.max()
        self.best_solution_ever = self.get_best_position()
        self.update_pheromone_matrix(self.population, self.fitnesses, True)

    def __str__(self):
        return 'ACO_TSP'

    def get_best_worst_fitnesses(self):
        return self.fitnesses.min(), self.fitnesses.max()
    
    def get_best_position(self):
        highest_fitness_index = np.argmax(self.fitnesses)
        return self.population[highest_fitness_index]
    
    def get_population(self):
        return self.population
    
    def get_fitnesses(self):
        return self.fitnesses
    
    def modify_population(self):
        raise NotImplementedError

    def update_pheromone_matrix_single_path(self, solution, fitness):
        for i in range(len(solution) - 1):
            self.pheromone_matrix[solution[i]][solution[i + 1]] += fitness

    def update_pheromone_matrix(self, solutions, fitnesses, first_update=False):
        self.pheromone_matrix *= self.rho
        if first_update:
            for i, solution in enumerate(solutions):
                self.update_pheromone_matrix_single_path(solution, fitnesses[i])
        else:
            if random.random() < self.update_with_best_ever_prob:
                self.update_pheromone_matrix_single_path(self.best_solution_ever,
                                                         self.best_fitness_ever)
            else:
                self.update_pheromone_matrix_single_path(self.get_best_position(),
                                                         self.get_best_worst_fitnesses()[1])

    def generate_solution_step(self, partial_solution, available_nodes):
        if len(available_nodes) == 1:
            partial_solution.append(available_nodes[0])
            return partial_solution
        else:
            current_node = partial_solution[-1]
            random_value = random.random()
            values_sum = 0
            for node in available_nodes:
                values_sum += math.pow(self.pheromone_matrix[current_node][node], self.alpha) * \
                                math.pow(self.problem.cost_matrix[current_node][node], self.beta)
            prob_sum = 0
            for node in available_nodes:
                if values_sum == 0:
                    prob_sum += 1 / len(available_nodes)
                else:
                    value = math.pow(self.pheromone_matrix[current_node][node], self.alpha) * \
                                    math.pow(self.problem.cost_matrix[current_node][node], self.beta)
                    prob_sum += value / values_sum
                if random_value < prob_sum:
                    partial_solution.append(node)
                    available_nodes.remove(node)
                    break
            return self.generate_solution_step(partial_solution, available_nodes)

    def generate_solution_iterative(self, partial_solution, available_nodes):
        while len(available_nodes) > 0:
            if len(available_nodes) == 1:
                partial_solution.append(available_nodes[0])
                available_nodes.remove(available_nodes[0])
            else:
                current_node = partial_solution[-1]
                random_value = random.random()
                values_sum = 0
                for node in available_nodes:
                    values_sum += math.pow(self.pheromone_matrix[current_node][node], self.alpha) * \
                                    math.pow(self.problem.cost_matrix[current_node][node], self.beta)
                prob_sum = 0
                for node in available_nodes:
                    if values_sum == 0:
                        prob_sum += 1 / len(available_nodes)
                    else:
                        value = math.pow(self.pheromone_matrix[current_node][node], self.alpha) * \
                                        math.pow(self.problem.cost_matrix[current_node][node], self.beta)
                        prob_sum += value / values_sum
                    if random_value < prob_sum:
                        partial_solution.append(node)
                        available_nodes.remove(node)
                        break
        return partial_solution

    def generate_solution(self):
        available_nodes = list(range(self.dimensions))
        sampled_node = random.sample(available_nodes, 1)[0]
        available_nodes.remove(sampled_node)
        partial_solution = [sampled_node]
        #solution = self.generate_solution_iterative(partial_solution, available_nodes)
        solution = self.generate_solution_step(partial_solution, available_nodes)
        return solution        

    def set_parameters(self, parameters):
        self.pop_size = int(parameters['pop_size'])
        self.alpha = parameters['alpha']
        self.beta = parameters['beta']
        self.rho = parameters['rho']
        self.update_with_best_ever_prob = parameters['update_with_best_ever_prob']
        self.v = parameters['v']

    def run(self, current_budget):
        self.population = np.array([self.generate_solution() for i in range(self.pop_size)])
        self.fitnesses = np.array([self.problem.evaluate(individual) for individual in self.population])
        best_fitness = self.fitnesses.max()
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_solution_ever = self.get_best_position()
        self.update_pheromone_matrix(self.population, self.fitnesses)
        return 1
