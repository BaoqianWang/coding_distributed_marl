import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.ticker import MaxNLocator


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()


labels = ['3 agents ', '8 agents']



# Every 100 iterations
Original = [20, 83]
Distributed = [15, 26]


x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(7.5,3.5))
rects2 = ax.bar(x -0.15, Original, width, label='Original')
rects3 = ax.bar(x , Distributed, width,label='Distributed')

# rects4 = ax.bar(x + 0.15, RandomLinear, width,label='Random sparse')
# rects5 = ax.bar(x + 0.3, Replication, width,label='Replication-based')
# rects6 = ax.bar(x + 0.45, LDPC, width,label='Regular LDPC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Training Time Over 100 Iterations (s)', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=20)
leg1 = ax.legend(loc='upper left', fontsize=19)

ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# Add second legend for the maxes and mins.
# leg1 will be removed from figure
# leg2 = ax.legend([rects2,rects3],['Original','Distributed'],loc='upper right',fontsize=19)
# # Manually add the first legend back
# ax.add_artist(leg1)


autolabel(rects2)
autolabel(rects3)


fig.tight_layout()

# plt.ylim((0, 15))

#plt.legend(fontsize=16)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.yticks(fontsize=20)
plt.grid()
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.9, top=0.86, wspace=None, hspace=None)
plt.savefig('original_distributed_comparison_spread.png', transparent = True)
plt.show()
