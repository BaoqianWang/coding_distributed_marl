import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.ticker import MaxNLocator


# predator prey environment: simple_tag
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

labels = ['0 straggler ', '5 straggler', '8stragglers']

# 8 agents
if(arglist.num_agents == 8):
    Original = [3.567,  0, 0]
    Uncoded = [2.52, 7.52, 7.54]
    MDS = [3.82, 3.80, 9.04]
    RandomLinear = [3.66, 3.73, 8.84]
    Replication = [2.91, 6.75, 8.16]
    LDPC = [2.65, 6.37, 7.90]


# 10 agents
if(arglist.num_agents == 10):
    Original = [5.26,  0, 0]
    Uncoded = [3.37, 8.34, 8.35]
    MDS = [5.48, 5.45, 10.66]
    RandomLinear = [5.07, 5.36, 10.34]
    Replication = [3.94, 9.12, 9.20]
    LDPC = [3.57, 7.96, 8.85]


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
    plt.ylim((0, 22))
else:
    plt.ylim((0, 17))
#plt.legend(fontsize=16)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.yticks(fontsize=20)
plt.grid()
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.9, top=0.96, wspace=None, hspace=None)
plt.savefig('adversary%d.png' %arglist.num_agents, transparent = True)
#plt.show()
