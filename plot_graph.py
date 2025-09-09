import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6,6))

# Cell boundary
cell = patches.Circle((0.5, 0.5), 0.45, facecolor="#e6f7ff", edgecolor="black")
ax.add_patch(cell)

# Nucleus
nucleus = patches.Circle((0.5, 0.5), 0.2, facecolor="#b3cde0", edgecolor="black")
ax.add_patch(nucleus)

# Ribosomes (dots)
for x, y in [(0.3,0.7),(0.7,0.3),(0.6,0.7),(0.4,0.3)]:
    ribo = patches.Circle((x,y), 0.02, facecolor="black")
    ax.add_patch(ribo)
    # Protein chain
    protein = patches.FancyArrow(x, y, 0.05, 0.05,
                                 width=0.005, color="green")
    ax.add_patch(protein)

# Style
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect('equal')
ax.axis('off')

plt.savefig("cell_diagram.png", dpi=300, bbox_inches="tight")
plt.show()
