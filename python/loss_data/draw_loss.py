import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20})

with open("train_loss_avgpool.txt") as f:
    loss_avgpool = f.readlines()
f.close()
with open("train_loss_maxpool.txt") as f:
    loss_maxpool = f.readlines()
f.close()
with open("acc_avgpool.txt") as f:
    acc_avgpool = f.readlines()
f.close()
with open("acc_maxpool.txt") as f:
    acc_maxpool = f.readlines()
f.close()

x = np.arange(0,len(loss_maxpool))

print(len(loss_avgpool), len(loss_maxpool), len(acc_avgpool), len(acc_maxpool))
before = np.array(loss_avgpool)
loss_avgpool = before.astype(np.float)

before = np.array(loss_maxpool)
loss_maxpool = before.astype(np.float)

before = np.array(acc_avgpool)
acc_avgpool = before.astype(np.float)

before = np.array(acc_maxpool)
acc_maxpool = before.astype(np.float)

fig, ax1 = plt.subplots()
#ax1.plot(x, loss_maxpool, 'b')
ax1.plot(x, loss_avgpool, 'b')
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
#ax2.plot(x, acc_maxpool, 'r')
ax2.plot(x, acc_avgpool, 'r')
ax2.set_ylabel('accuracy', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()