import os
import json
import random
import numpy as np
from scipy.spatial import distance
from cec17_functions import cec17_test_func

class CEC17:

    def __init__(self, cec17_config_json):
        with open(cec17_config_json, 'r') as f:
            json_data = json.load(f)
            self.training_instances = json_data['training_functions']
            self.test_instances = json_data['test_functions']
        self.running_instances = self.training_instances
        self.dll_path = os.path.abspath('cec17_test_func.so')
        self.indexes = []
        self.next_instance()


    def __str__(self):
        return 'func' + str(self.func_num) + '_dimensions' + str(self.dimensions)


    def next_instance(self):
        if len(self.indexes) == 0:
            self.indexes = random.sample(range(len(self.running_instances)), len(self.running_instances))
        new_index = self.indexes.pop()
        self.dimensions = self.running_instances[new_index]['dimensions']
        self.func_num = self.running_instances[new_index]['func_num']


    def get_dimensions(self):
        return self.dimensions


    def evaluate(self, solution):
        f = [0]
        if self.dll_path is None:
            cec17_test_func(solution, f, self.dimensions, 1, self.func_num)
        else:
            cec17_test_func(solution, f, self.dimensions, 1, self.func_num, self.dll_path)
        return -f[0]


    def search_space_bounds(self):
        return [(-100, 100)] * self.dimensions


    def get_optimum(self):
        return -self.func_num * 100


    def distance(self, vector1, vector2):
        return distance.euclidean(vector1, vector2)


    def max_distance(self):
        return distance.euclidean(np.full(self.dimensions, -100), np.full(self.dimensions, 100)) 