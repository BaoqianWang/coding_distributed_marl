import numpy as np
import math
import random
import time
from itertools import combinations
from numpy.linalg import matrix_rank
import copy

random.seed(30)
np.random.seed(30)


class MDS_code():
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.generate_encode_matrix()
        self.weight_key = ['p_variables','target_p_variables','q_variables','target_q_variables']

    def generate_encode_matrix(self):
        x = np.linspace(.1,.5,self.m)

        # Generate Vandermonde matrix for encoding
        #self.H = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1], [1,1,1]])#np.vander(x, self.n).transpose()
        self.H = np.vander(x, self.n).transpose()

    def decode(self, received_data, weight_length):
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
                    sum_weights.append(weight)
                sum_weights_dict[key] = copy.deepcopy(sum_weights)

            agent_weight[j] = copy.deepcopy(sum_weights_dict)

        return agent_weight

if __name__=='__main__':
    mds_code = MDS_code(9, 10)
    print(mds_code.H)
    print(mds_code.H[0])
    # for row in mds_code.H:
    #     for a in row:
    #         print(a)
