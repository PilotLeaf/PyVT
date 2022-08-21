import pandas as pd
import matplotlib.pyplot as plt
from pyais import traj_segment


def main():
    trajdata = pd.read_csv('../data/1.csv', encoding="gbk",
                           usecols=['DRMMSI', 'DRGPSTIME', 'DRLONGITUDE', 'DRLATITUDE'])
    scattercolors = ['blue', 'red', 'yellow', 'cyan', 'purple', 'orange', 'olive', 'brown', 'black', 'm']
    rawgrouped = trajdata[:].groupby('DRMMSI')
    for name, group in rawgrouped:
        plt.plot(group['DRLONGITUDE'], group['DRLATITUDE'], marker='*', ms=8, linestyle='--',
                 color='blue', linewidth=0.75)
    segmenteddata = traj_segment.segment(trajdata, 1500)
    seggrouped = segmenteddata[:].groupby('DRMMSI')
    i = 0
    for name, group in seggrouped:
        i = i + 1
        colorSytle = scattercolors[i % len(scattercolors)]
        plt.plot(group['DRLONGITUDE'], group['DRLATITUDE'], marker='o', ms=5, linestyle='-',
                 color=colorSytle, linewidth=0.5)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.title('Trajectory Segmentation', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
