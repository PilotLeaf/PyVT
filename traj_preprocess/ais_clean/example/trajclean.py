import pandas as pd
import matplotlib.pyplot as plt
from pyais import traj_clean


def plot(rawdf, cleandf, rl):
    fig, ax = plt.subplots()
    for shipmmsi, dt in rawdf.groupby('DRMMSI'):
        if len(dt) >= rl and len(str(shipmmsi)) > 5:
            dt.sort_values(by='DRGPSTIME', ascending=True, inplace=True)
            ax.plot(dt.loc[:, 'DRLONGITUDE'].values, dt.loc[:, 'DRLATITUDE'].values, marker='*', linestyle='--',
                    color='red',
                    linewidth=0.5)
    for shipmmsi, dt in cleandf.groupby('DRMMSI'):
        if len(dt) >= rl and len(str(shipmmsi)) > 5:
            dt.sort_values(by='DRGPSTIME', ascending=True, inplace=True)
            ax.plot(dt.loc[:, 'DRLONGITUDE'].values, dt.loc[:, 'DRLATITUDE'].values, marker='>', color='green',
                    linewidth=0.75)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.title('Trajectory Cleaning', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    rawdf = pd.read_csv('../data/1.csv')
    cleandf = traj_clean.heuristic_clean(rawdf, 25)
    plot(rawdf, cleandf, 1)
