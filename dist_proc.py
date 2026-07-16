import numpy as np
import numpy.linalg as lin
import scipy as sci
import math

"""
get_spec_vector takes a dictionary as input and creates an array of it's values
"""


def get_spec_vector(dict):
    vector = []
    for i in dict:
        vector.append(dict[i])
    return vector


"""
diff list takes two lists and performs the element-wise substraction, boolean absolute is for absolute difference value 
"""


def list_diff(list_1, list_2, absolute):
    if len(list_1) != len(list_2):
        raise ValueError("Len 1 != Len 2")
    else:
        i = 0
        diff_list = []
        for i in range(len(list_1)):
            a = list_1[i]
            b = list_2[i]
            if absolute == True:
                diff = abs(a - b)
            else:
                diff = a - b
            diff_list.append(diff)
    return diff_list


"""
get_spec_distance takes two lists as input and via list_diff() calculates a length of list1 - list2 
"""


def spec_distance(spec_from, spec_to):
    diffs = list_diff(spec_from, spec_to, True)
    i = 0
    while i < len(diffs):
        a = diffs[i]
        a = a * a
        diffs[i] = a
        i += 1
    b = 0
    for i in diffs:
        b += i
    dist = math.sqrt(b)
    return dist


def dist_matrix(*args):
    dist_matrix = []
    for n in range(0, len(args)):
        line = []
        for m in range(0, len(args)):
            line.append(spec_distance(args[m], args[n]))
        dist_matrix.append(line)
    return dist_matrix
