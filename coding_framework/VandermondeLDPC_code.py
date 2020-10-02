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
from sympy import *
from gf2elim import gf2elim


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

        self.H = np.zeros([self.p*self.gamma, self.p*self.pho])
        for i in range(self.gamma):
            for j in range(self.pho):
                self.H[i*self.p:(i+1)*self.p, j*self.p:(j+1)*self.p] = matrix_power(A, i*j)


        H_new, G_systematic =coding_matrix_systematic(self.H)
        self.G = G_systematic
        return





if __name__=='__main__':
    regular_ldpc_code = VandermondeLDPC_code(2, 2, 1)
