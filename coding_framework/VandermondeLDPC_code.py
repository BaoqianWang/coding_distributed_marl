import numpy as np
import math
import random
import time
import copy
from itertools import combinations
from numpy.linalg import matrix_rank
from pyldpc import make_ldpc, encode, decode, get_message, coding_matrix_systematic
import copy
from itertools import permutations
from numpy.linalg import matrix_power
from numpy.linalg import inv



random.seed(30)
np.random.seed(30)


class VandermondeLDPC_code():
    def __init__(self, p, pho, gamma):
        # Regular LDPC (n, d_v, d_c)
        # d_c is the number of non-zeros in each row
        # d_v is the number of non-zeros in each column
        self.p = p
        self.pho = pho
        self.gamma = gamma
        ##In the replication based scheme, we assume n/m is positive integer
        self.get_generator_matrix()
        #self.get_generator_matrix()
        self.weight_key = ['p_variables','target_p_variables','q_variables','target_q_variables']


    def get_generator_matrix(self):
        #I_matrix = np.identity(self.p)
        A = np.identity(self.p-1)
        A_column = np.zeros([self.p, 1])
        A_column[-1, 0] = 1
        A_row = np.zeros([1, self.p-1])
        A = np.vstack((A, A_row))
        A = np.hstack((A_column, A))
        # print(A)

        self.Pa = np.zeros([self.p*self.gamma, self.p*self.pho])
        for i in range(self.gamma):
            for j in range(self.pho):
                self.Pa[i*self.p:(i+1)*self.p, j*self.p:(j+1)*self.p] = matrix_power(A, i*j)


        H_new, G_systematic =coding_matrix_systematic(self.Pa)
        self.H = G_systematic
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


        return

    def decode(self, received_data, weight_length, weight_shape):
        self.variable_indexes = copy.deepcopy(self.backup_variable_indexes)
        self.variable_degrees = copy.deepcopy(self.backup_variable_degrees)
        self.check_indexes = copy.deepcopy(self.backup_check_indexes)
        received_index = [data[0] for data in received_data]
        print(received_index)
        for index in self.check_indexes:
            temp_index = copy.deepcopy(index)
            for item in temp_index:
                if item not in received_index:
                    index.remove(item)

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
                    print(variable_index[0])
                    for item in self.check_indexes[variable_index[0]]:
                        #print('item, check_indexes', item, self.check_indexes)
                        if len(self.variable_indexes[item])>1:
                            self.variable_indexes[item].remove(variable_index[0])
                            for key in self.weight_key:
                                for k in range(weight_length):
                                    #print(received_index)
                                    item_index = received_index.index(item)
                                    received_data[item_index][1][key][k] -=  data[1][key][k]
        print('Done')
        for j in range(self.m):
            for key in self.weight_key:
                for k in range(weight_length):
                    agent_weight[j][key][k].resize(weight_shape[j][key][k])


        return agent_weight


if __name__=='__main__':
    #regular_ldpc_code = VandermondeLDPC_code(3, 3, 2)
    regular_ldpc_code = VandermondeLDPC_code(2, 3, 2)
    print(regular_ldpc_code.H)
    print(regular_ldpc_code.H.shape[1])
