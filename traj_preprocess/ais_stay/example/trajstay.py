import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyais import traj_stay


def stay_st_test(traj_file: str):
    params = {'axes.titlesize': 'large',
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'legend.fontsize': 16,
              'legend.handlelength': 3}
    plt.rcParams.update(params)

    trajdata = traj_stay.stay_st_detect(traj_file)
    scatterColors = ['blue', 'red', 'yellow', 'cyan', 'purple', 'orange', 'olive', 'brown', 'black', 'm']
    for shipmmsi, dt in trajdata.groupby('DRMMSI'):
        labels = dt['SP_Status'].values
        clusterNum = len(set(labels))
        plt.plot(dt['DRLONGITUDE'].values, dt['DRLATITUDE'].values, marker='o', markeredgewidth=1.0, linewidth=0.75,
                 markerfacecolor='white', markeredgecolor='m', ms=4, alpha=1.0, color='m', zorder=2)
        for i in range(-1, clusterNum):
            colorSytle = scatterColors[i % len(scatterColors)]
            subCluster = dt.query('SP_Status == @i')
            if i >= 0:
                plt.scatter(subCluster['DRLONGITUDE'].values, subCluster['DRLATITUDE'].values, marker='*', s=70,
                            edgecolors=colorSytle,
                            c='white', linewidths=1.0, zorder=3)
            else:
                continue
    plt.title('Stay Point Identification', fontsize=16)
    plt.xlabel('Longitude', fontsize=16)
    plt.ylabel('Latitude', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def stay_spt_test(traj_file: str):
    params = {'axes.titlesize': 'large',
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'legend.fontsize': 14,
              'legend.handlelength': 3}
    plt.rcParams.update(params)

    trajdata = traj_stay.stay_spt_detect(traj_file)
    scatterColors = ['blue', 'red', 'yellow', 'cyan', 'm']
    for shipmmsi, dt in trajdata.groupby('DRMMSI'):
        plt.plot(dt['DRLONGITUDE'].values, dt['DRLATITUDE'].values, marker='o', markeredgewidth=1.0, linewidth=0.75,
                 markerfacecolor='white', markeredgecolor='m', ms=4, alpha=1.0, color='m', zorder=2)
        lbs = set(dt['SP_Status'].values)
        for index, v in enumerate(lbs):
            if v > 0:
                colorSytle = scatterColors[index % len(scatterColors)]
                subCluster = dt.query('SP_Status == @v')
                plt.scatter(subCluster['DRLONGITUDE'].values, subCluster['DRLATITUDE'].values, marker='*', s=70,
                            edgecolors=colorSytle,
                            c='white', linewidths=1.0, zorder=3)

    plt.title('Stay Point Identification', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # stay_spt_test("../data/1.csv")
    stay_st_test("../data/1.csv")
