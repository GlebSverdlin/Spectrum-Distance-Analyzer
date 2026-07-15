import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy as sci
import math
from matplotlib import ticker


"""
example dataset: 3 stars, 4 elements each
"""

a_ori = {"Ha": 56, "Hb": 43, "O3": 21, "N2": 36}
b_ori = {"Ha": 61, "Hb": 36, "O3": 27, "N2": 47}
t_ori = {"Ha": 32, "Hb": 33, "O3": 42, "N2": 49}

def get_spec_vector(list):
    vector = []
    for i in list:
        vector.append(list[i])
    return vector

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
        

a_ori_spec=get_spec_vector(a_ori)
b_ori_spec=get_spec_vector(b_ori)
t_ori_spec=get_spec_vector(t_ori)

def get_spec_distance(spec_1, spec_2):
    diffs = list_diff(spec_1, spec_2, True)
    i = 0
    while i<len(diffs):
        a = diffs[i]
        a = a*a
        diffs[i]=a
        i+=1
    b=0
    for i in diffs:
        b+=i
    b=math.sqrt(b)
    return b

fig, ax = plt.subplots()

ab = get_spec_distance(a_ori_spec, b_ori_spec)
at = get_spec_distance(a_ori_spec, t_ori_spec)
bt = get_spec_distance(b_ori_spec, t_ori_spec)

stars= ['Alpha', 'Beta', 'Tau']

vals = np.array([[0, ab, at],
                 [ab, 0, bt],
                 [at, bt, 0]])

ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

ax.set_xticks(range(len(stars)), labels=stars)
ax.set_yticks(range(len(stars)), labels=stars)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.imshow(vals, cmap='coolwarm')
plt.colorbar(label='Distance')
for i in range(len(stars)):
    for j in range(len(stars)):
        dist = vals[i,j]
        dist=round(dist, 1)
        text = ax.text(j, i, dist,
                       ha="center", va="center", color="w")
#fig.savefig("figure.pdf")
plt.show()
print(ab, at, bt)
