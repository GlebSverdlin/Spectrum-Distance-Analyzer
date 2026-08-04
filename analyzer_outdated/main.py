import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy as sci
import math
from matplotlib import ticker
from dist_proc import *
from matrix_functions import *

a_ori = {"Ha": 56, "Hb": 43, "O3": 21, "N2": 36}
b_ori = {"Ha": 61, "Hb": 36, "O3": 27, "N2": 47}
t_ori = {"Ha": 32, "Hb": 33, "O3": 42, "N2": 49}


vect_a = get_spec_vector(a_ori)
vect_b = get_spec_vector(b_ori)
vect_t = get_spec_vector(t_ori)
vect_g = [24, 74, 21, 56]

spec_list = [vect_a, vect_b, vect_t, vect_g]

matrix = dist_matrix(spec_list)

matrix_print(matrix)


def call_dist_matrix():
    return matrix
