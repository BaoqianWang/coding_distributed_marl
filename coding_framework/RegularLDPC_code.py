import numpy as np
import math
import random
import time
import copy
from itertools import combinations
from numpy.linalg import matrix_rank
from pyldpc import make_ldpc, encode, decode, get_message
import copy


random.seed(30)
np.random.seed(30)



class RegularLDPC_code():
    def __init__(self, n, dv, dc):
        self.n = n
        self.dv = dv
        self.dc = dc
        ##In the replication based scheme, we assume n/m is positive integer
        self.generate_encode_matrix()
        self.weight_key = ['p_variables','target_p_variables','q_variables','target_q_variables']


    def generate_encode_matrix(self):
        self.Pa , self.H =  make_ldpc(self.n, self.dv, self.dc, systematic=True, sparse=True)
        self.m = self.H.shape[1]
        self.variable_indexes = []
        self.variable_degrees = []

        self.check_indexes = []

        for each_row in self.H:
            degree = 0
            index =[]
            for i, entry in enumerate(each_row):
                if(entry==1):
                    degree+=1
                    index.append(i)
            self.variable_indexes.append(index)
            self.variable_degrees.append(degree)


        for each_column in self.H.transpose():
            column_index = []
            for i, entry in enumerate(each_column):
                if(entry ==1):
                    column_index.append(i)

            self.check_indexes.append(column_index)



        self.backup_variable_indexes = copy.deepcopy(self.variable_indexes)
        self.backup_variable_degrees = copy.deepcopy(self.variable_degrees)
        self.backup_check_indexes = copy.deepcopy(self.check_indexes)

    # def decode(self, received_data, weight_length):
    #     index_all = [data[0] for data in received_data]
    #
    #     sub_index_combination = combinations(index_all, self.m)
    #     for sub_index in sub_index_combination:
    #         if(matrix_rank(self.H[sub_index,:])==self.m):
    #             self.decode_H = np.linalg.inv(self.H[sub_index,:])
    #             encoded_weight=[received_data[index_all.index(j)][1] for j in sub_index]
    #             break
    #
    #
    #     agent_weight = [None]*self.m
    #
    #     for j in range(self.m):
    #         sum_weights=[]
    #         for k in range(weight_length):
    #             weight = 0
    #             for i, entry in enumerate(self.decode_H[j]):
    #                 weight += entry*encoded_weight[i][k]
    #             sum_weights.append(weight)
    #         agent_weight[j] = sum_weights
    #
    #     return agent_weight

    def decode(self, received_data, weight_length):

        self.variable_indexes = copy.deepcopy(self.backup_variable_indexes)
        self.variable_degrees = copy.deepcopy(self.backup_variable_degrees)
        self.check_indexes = copy.deepcopy(self.backup_check_indexes)
        received_index = [data[0] for data in received_data]
        #print('H', self.H)
        #print('Original check indexes', self.check_indexes)
        #print('Received index', received_index)
        #print(received_index)
        #print('check indexes', self.check_indexes)
        for index in self.check_indexes:
            #print('index is ', index)
            temp_index = copy.deepcopy(index)
            for item in temp_index:
                #print(item)
                if item not in received_index:
                    #print('remove',item)
                    #print('Before remove', index)
                    index.remove(item)
                    #print('After remove', index)
        #print(self.check_indexes)
        #print('After check indexes', self.check_indexes)

        agent_weight = [None]*self.m
        iteration = 0
        while None in agent_weight:
            #print('iteration', iteration)
            iteration += 1

            for i, data in enumerate(received_data):
                #print('learner index', data[0])
                variable_index = copy.deepcopy(self.variable_indexes[data[0]])
                #print('variable_index', variable_index)
                if len(variable_index) == 1 and agent_weight[variable_index[0]] is None:
                    agent_weight[variable_index[0]] = copy.deepcopy(data[1])

                    for item in self.check_indexes[variable_index[0]]:
                        #print('item, check_indexes', item, self.check_indexes)
                        if len(self.variable_indexes[item])>1:
                            self.variable_indexes[item].remove(variable_index[0])
                            for key in self.weight_key:
                                for k in range(weight_length):
                                    #print(received_index)
                                    item_index = received_index.index(item)
                                    received_data[item_index][1][key][k] -=  data[1][key][k]

        return agent_weight

if __name__=='__main__':
    # Regular LDPC (n, d_v, d_c)
    # d_c is the number of non-zeros in each row
    # d_v is the number of non-zeros in each column
    regular_ldpc_code = RegularLDPC_code(10, 2, 5)


    print(regular_ldpc_code.H)
    print(regular_ldpc_code.Pa)
    #for row in regular_ldpc_code.H.transpose():
        #print(row)

    #for column in regular_ldpc_code.H:
        #print(column)




         #for a in row:
             #print(a)
