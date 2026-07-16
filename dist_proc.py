import numpy as np
import numpy.linalg as lin
import scipy as sci
import math

'''
get_spec_vector takes a dictionary as input and creates an array of it's values
'''
def get_spec_vector(dict):
    vector = []
    for i in dict:
        vector.append(dict[i])
    return vector


'''
diff list takes two lists and performs the element-wise substraction, boolean absolute is for absolute difference value 
'''
def list_diff(list_1, list_2, absolute):
    if len(list_1)!=len(list_2):
        raise ValueError('Len 1 != Len 2')
    else:
        i=0
        diff_list = []
        for i in range(len(list_1)):
            a = list_1[i]
            b = list_2[i]
            if absolute == True: diff = abs(a-b)
            else: diff = a-b
            diff_list.append(diff)
    return diff_list


def get_spec_distance(spec_from, spec_to):
    diffs = list_diff(spec_from, spec_to, True)
    i = 0
    while i<len(diffs):
        a = diffs[i]
        a = a*a
        diffs[i]=a
        i+=1
    b=0
    for i in diffs:
        b+=i
    dist=math.sqrt(b)
    return dist


class spec_distance:
    int dist
    def __init__(self, from_obj, to_obj): #и from и to - списки величин потоков для соотв. звезд
        self.spec_from = from_obj
        self.spec_to = to_obj
        dist = get_spec_distance(spec_from, spec_to)
