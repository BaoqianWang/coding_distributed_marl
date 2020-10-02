import numpy as np
import math
import random
import time
import copy
from itertools import combinations
from numpy.linalg import matrix_rank
import copy

random.seed(30)
np.random.seed(30)



class ReplicationBased():
    def __init__(self, m, n):
        self.m = m
        self.n = n
        ##In the replication based scheme, we assume n/m is positive integer
        self.generate_encode_matrix()
        self.weight_key = ['p_variables','target_p_variables','q_variables','target_q_variables']


    def generate_encode_matrix(self):
        m_identity = np.identity(self.m)
        self.H = copy.deepcopy(m_identity)
        for i in range(self.n-self.m):
            temp = np.zeros((1, self.m))
            temp[0][i] = 1
            self.H=np.vstack((self.H, temp))

        # for i in range(int(self.n/self.m)-1):
        #     self.H=np.vstack((self.H, m_identity))


    def decode(self, received_data, weight_length):
        index_all = [data[0] for data in received_data]

        sub_index_combination = combinations(index_all, self.m)
        for sub_index in sub_index_combination:
            if(matrix_rank(self.H[sub_index,:])==self.m):
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
    mds_code = ReplicationBased(5, 8)
    print(mds_code.H)
    # for row in mds_code.H:
    #     for a in row:
    #         print(a)
