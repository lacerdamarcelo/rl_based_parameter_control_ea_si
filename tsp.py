import json
import random
import numpy as np

class TSP:

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
        return 'instance_file_path_' + str(self.instance_file_path)


    def next_instance(self):
        if len(self.indexes) == 0:
            self.indexes = random.sample(range(len(self.running_instances)), len(self.running_instances))
        new_index = self.indexes.pop()
        running_instance = self.running_instances[new_index]
        #print(running_instance)
        self.instance_file_path = running_instance['instance_file_path']
        with open(self.instances_path + '/' + self.instance_file_path, 'r') as f:
            lines = f.readlines()
            self.num_cities = int(lines[5].split(' : ')[1].split('\n')[0])
            self.cost_matrix = np.zeros((self.num_cities, self.num_cities))
            nodes = np.zeros((self.num_cities, 2))
            for i, line in enumerate(lines[8:-2]):
                line_spt = line.split(' ')
                x = int(line_spt[1])
                y = int(line_spt[2].split('\n')[0])
                nodes[i] = (x,y)
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i != j:
                        self.cost_matrix[i][j] = np.sqrt(np.power(nodes[i][0] - nodes[j][0], 2) + \
                            np.power(nodes[i][1] - nodes[j][1], 2))


    def get_dimensions(self):
        return self.num_cities


    def evaluate(self, solution):
        cost = 0
        for i in range(self.num_cities - 1):
            cost += self.cost_matrix[solution[i] - 1][solution[i + 1] - 1]
        return 1 / cost


    def distance(self, vector1, vector2):
        return (np.array(vector1) == np.array(vector2)).sum()


    def max_distance(self):
        return self.num_cities


if __name__ == '__main__':
    tsp = TSP('tsp_config.json')
    print(tsp.cost_matrix)
    #print(tsp.evaluate(random.sample(range(1, 1001), 1000)))
