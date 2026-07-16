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

matrix_print(dist_matrix(vect_a, vect_b, vect_t))




"""
example dataset: 3 stars, 4 elements each
"""




























'''

fig, ax = plt.subplots()

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

a_ori_spec=get_spec_vector(a_ori)
b_ori_spec=get_spec_vector(b_ori)
t_ori_spec=get_spec_vector(t_ori)

'''
