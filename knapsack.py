import json
import random
import numpy as np

class Knapsack:

    def __init__(self, config_json):
        with open(config_json, 'r') as f:
            json_data = json.load(f)
            self.training_instances = json_data['training_functions']
            self.test_instances = json_data['test_functions']
            self.instances_path = json_data['instances_path']
        self.running_instances = self.training_instances
        self.indexes = []
        self.next_instance()

    
    def __str__(self):
        return 'instance_type' + str(self.instance_type) + '_num_items' + str(self.num_items) + '_max_coefficient' + str(self.max_coefficient) + '_instance' + str(self.instance_index)


    def next_instance(self):
        if len(self.indexes) == 0:
            self.indexes = random.sample(range(len(self.running_instances)), len(self.running_instances))
        new_index = self.indexes.pop()
        running_instance = self.running_instances[new_index]
        self.instance_type = running_instance['instance_type']
        self.num_items = running_instance['num_items']
        self.max_coefficient = 1000
        with open(self.instances_path + '/knapPI_' + str(self.instance_type) + '_' + str(self.num_items) + \
                  '_' + str(self.max_coefficient) + '.csv', 'r') as f:
            self.instance_index = running_instance['instance_index']
            file_data = f.readlines()
            pointer = 0
            scenario = {'data': []}
            get_data = False
            self.items = []
            for line in file_data:
                if line == 'knapPI_' + str(self.instance_type) + '_' + str(self.num_items) + '_' + \
                        str(self.max_coefficient) + '_' + str(self.instance_index) + '\n':
                    get_data = True
                if get_data:
                    if line == '-----\n':
                        break
                    else:
                        if pointer == 2:
                            line_spt = line.split(' ')
                            self.scenario_capacity = int(line_spt[1].split('\n')[0])
                        elif pointer >= 5:
                            line_spt = line.split(',')
                            self.items.append({'value': int(line_spt[1]), 'weight': int(line_spt[2])})
                    pointer += 1


    def get_dimensions(self):
        return len(self.items)


    def evaluate(self, solution):
        sum_values = 0
        sum_weights = 0
        for i, value in enumerate(solution):
            sum_values += self.items[i]['value'] * value
            sum_weights += self.items[i]['weight'] * value
        return sum_values + 0.00001 if sum_weights <= self.scenario_capacity else 0.00001


    def evaluate_batch(self, solutions):
        values = []
        for solution in solutions:
            sum_values = 0
            sum_weights = 0
            for i, value in enumerate(solution):
                    sum_values += self.items[i]['value'] * value
                    sum_weights += self.items[i]['weight'] * value
            value = sum_values + 0.00001 if sum_weights <= self.scenario_capacity else 0.00001
            values.append(value)
        return np.array(values)


    def distance(self, vector1, vector2):
        return (np.array(vector1) - np.array(vector2)).sum()


    def max_distance(self):
        return len(self.items)


if __name__ == '__main__':
    knp = Knapsack("knapsack_config.json")
    test = np.random.randint(0, 2, 200)
    knp.evaluate(test)
    print(knp.items)
