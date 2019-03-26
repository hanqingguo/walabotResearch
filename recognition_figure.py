import numpy as np
import matplotlib
import matplotlib.pyplot as plt

true = ["fall", "stand-up", "sit-to-stand", "stand-to-sit","walk",
              "still"]
predict = ["fall", "stand-up", "sit-to-stand", "stand-to-sit","walk",
              "still"]



# length = 3
# harvest = np.array([[31, 9, 15, 28, 10, 0],
#                     [1, 37, 18, 14, 9, 0],
#                     [3, 29, 38, 5, 17, 1],
#                     [15, 1, 0, 48, 17, 1],
#                     [3, 5, 19, 25, 44, 0],
#                     [0, 0, 0, 1, 2, 78],
#                     ])

# length = 5
# harvest = np.array([[83, 2, 1, 3, 4, 0],
#                     [0, 72, 3, 1, 4, 0],
#                     [2, 2, 73, 1, 14, 1],
#                     [6, 1, 0, 66, 8, 1],
#                     [1, 2, 11, 3, 79, 0],
#                     [0, 0, 0, 0, 0, 81],
#                     ])

length = 8
harvest = np.array([[85, 1, 2, 2, 4, 0],
                    [0, 77, 1, 0, 1, 0],
                    [18, 28, 37, 5, 5, 0],
                    [29, 12, 4, 35, 1, 1],
                    [17, 22, 6, 7, 44, 0],
                    [0, 0, 0, 0, 0, 81],
                    ])


fig, ax = plt.subplots()
im = ax.imshow(harvest, interpolation="nearest")

# We want to show all ticks...
ax.set_xticks(np.arange(len(predict)))
ax.set_yticks(np.arange(len(true)))
# ... and label them with the respective list entries
ax.set_xticklabels(predict)
ax.set_yticklabels(true)
ax.set_xlabel("Predict", fontsize=12, color='green')
ax.set_ylabel("Ground Truth", fontsize=12, color='green')
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

fig.colorbar(im)
# Loop over data dimensions and create text annotations.
for i in range(len(true)):
    for j in range(len(predict)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="r")

ax.set_title("Confusion Matrix")
fig.tight_layout()
plt.show()

