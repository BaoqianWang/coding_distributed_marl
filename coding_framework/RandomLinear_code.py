import numpy as np
import math
import random
import time
from itertools import combinations
from numpy.linalg import matrix_rank
import copy
random.seed(30)
np.random.seed(30)



class RandomLinear_code():
    def __init__(self, m, n, pk):
        self.m = m
        self.n = n
        self.pk = pk
        self.generate_encode_matrix()
        self.weight_key = ['p_variables','target_p_variables','q_variables','target_q_variables']


    def generate_encode_matrix(self):
        self.H=np.random.normal(0, 1, size=(self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                if random.random() < 1-self.pk:
                    self.H[i,j]=0

    def decode(self, received_data, weight_length, weight_shape):
        index_all = [data[0] for data in received_data]

        sub_index_combination = combinations(index_all, self.m)
        for sub_index in sub_index_combination:
            if(matrix_rank(self.H[sub_index,:]) == self.m):
                self.decode_H = np.linalg.inv(self.H[sub_index,:])
                encoded_weight=[received_data[index_all.index(j)][1] for j in sub_index]
                break


        agent_weight = [None]*self.m

        for j in range(self.m):
            sum_weights_dict = dict()
            for key in self.weight_key:
                sum_weights=[]
                for k in range(weight_length):
                    weight = 0
                    for i, entry in enumerate(self.decode_H[j]):
                        weight += entry*encoded_weight[i][key][k]
                    weight.resize(weight_shape[j][key][k])
                    sum_weights.append(weight)
                sum_weights_dict[key] = copy.deepcopy(sum_weights)

            agent_weight[j] = copy.deepcopy(sum_weights_dict)

        return agent_weight

if __name__=='__main__':
    mds_code = RandomLinear_code(9, 10, 0.7)
    print(mds_code.H)
    #print(mds_code.H[0])
    # for row in mds_code.H:
    #     for a in row:
    #         print(a)
