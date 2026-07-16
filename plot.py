import matplotlib.pyplot as plt
from matplotlib import ticker
from main import *

fig, ax = plt.subplots()

stars = ["Alpha", "Beta", "Tau", "Gamma"]

vals = call_dist_matrix()

ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

ax.set_xticks(range(len(stars)), labels=stars)
ax.set_yticks(range(len(stars)), labels=stars)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.imshow(vals, cmap="plasma_r")
plt.colorbar(label="Distance")
for i in range(len(stars)):
    for j in range(len(stars)):
        dist = vals[i][j]
        dist = round(dist, 1)
        text = ax.text(j, i, dist, ha="center", va="center", color="w")
# fig.savefig("figure.pdf")
plt.show()
