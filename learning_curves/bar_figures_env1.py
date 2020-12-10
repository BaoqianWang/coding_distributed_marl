import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.ticker import MaxNLocator

def parse_args():
    parser = argparse.ArgumentParser("Figure comparison")
    # Environment
    parser.add_argument("--num_agents", type=int, default="8", help="num agents")


    return parser.parse_args()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()

arglist = parse_args()
labels = ['0 straggler ', '1 straggler', '2 stragglers']

# 8 agents
if(arglist.num_agents == 8):
    Original = [5.64,  0, 0]
    Uncoded = [4.67, 5.73, 5.74]
    MDS = [6.281, 6.286, 6.279]
    RandomLinear = [6.09, 6.099, 6.10]
    Replication = [5.17, 5.20, 5.8]
    LDPC = [4.92, 4.95, 5.3]


# 10 agents
if(arglist.num_agents == 10):
    Original = [9.22,  0, 0]
    Uncoded = [7.31, 8.58, 8.60]
    MDS = [10.09, 10.35,10.35]
    RandomLinear = [9.825, 9.81, 9.83]
    Replication = [8.389, 8.60, 9.2]
    LDPC = [7.76, 8.03, 8.45]


x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(7.5,3.5))
#plt.figure(figsize=(7,4.3))
#rects1 = ax.bar(x-0.3 , Original, width, label='Original MADDPG')
rects2 = ax.bar(x -0.15, Uncoded, width, label='Uncoded')
rects3 = ax.bar(x , MDS, width,label='MDS')
rects4 = ax.bar(x + 0.15, RandomLinear, width)
rects5 = ax.bar(x + 0.3, Replication, width)
rects6 = ax.bar(x + 0.45, LDPC, width)
# rects4 = ax.bar(x + 0.15, RandomLinear, width,label='Random sparse')
# rects5 = ax.bar(x + 0.3, Replication, width,label='Replication-based')
# rects6 = ax.bar(x + 0.45, LDPC, width,label='Regular LDPC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (s)', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=20)
leg1 = ax.legend(loc='upper left', fontsize=19)

ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# Add second legend for the maxes and mins.
# leg1 will be removed from figure
leg2 = ax.legend([rects4,rects5,rects6],['Random sparse','Replication-based','Regular LDPC'],loc='upper right',fontsize=19)
# Manually add the first legend back
ax.add_artist(leg1)






#autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

fig.tight_layout()
if(arglist.num_agents == 10):
    plt.ylim((0, 20))
else:
    plt.ylim((0, 15))

#plt.legend(fontsize=16)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.yticks(fontsize=20)
plt.grid()
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.9, top=0.96, wspace=None, hspace=None)
plt.savefig('spread%d.png' %arglist.num_agents, transparent = True)
#plt.show()
