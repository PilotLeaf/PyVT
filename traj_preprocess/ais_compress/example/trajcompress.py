import pandas as pd
import matplotlib.pyplot as plt
from pyais import traj_compress
from pyais.traj_compress import douglaspeucker


def sbctest():
    trajdata = pd.read_csv('../data/1.csv', encoding="gbk")
    compresseddata = traj_compress.sbc(trajdata, 0.025, 0.25)
    plt.plot(trajdata['DRLONGITUDE'], trajdata['DRLATITUDE'], marker='*', linestyle='--', color='blue', linewidth=0.5,
             label='Raw trajectory')
    plt.plot(compresseddata['DRLONGITUDE'], compresseddata['DRLATITUDE'], marker='o', ms=5, linestyle='-', color='g',
             linewidth=0.5, label='Compressed trajectory')
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.title('Trajectory Compression', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def dptest():
    data = pd.read_csv('../data/1.csv', usecols=['DRMMSI', 'DRGPSTIME', 'DRLONGITUDE', 'DRLATITUDE'])
    grouped = data[:].groupby('DRMMSI')
    for name, group in grouped:
        trajdata = group.sort_values(by='DRGPSTIME', ascending=True)
        dp = douglaspeucker(trajdata[['DRLATITUDE', 'DRLONGITUDE']].values)
        epsilon = dp.avg()
        mask = dp.rdp(group[['DRLATITUDE', 'DRLONGITUDE']].values, epsilon, algo="iter", return_mask=True)
        compresseddata = group[mask]
        plt.plot(trajdata['DRLONGITUDE'], trajdata['DRLATITUDE'], marker='*', linestyle='--', color='blue',
                 linewidth=0.5,
                 label='Raw trajectory')
        plt.plot(compresseddata['DRLONGITUDE'], compresseddata['DRLATITUDE'], marker='o', ms=5, linestyle='-',
                 color='g',
                 linewidth=0.5, label='Compressed trajectory')
        plt.yticks(fontproperties='Times New Roman', size=14)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 16})
        plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 16})
        plt.title('Trajectory Compression', fontdict={'family': 'Times New Roman', 'size': 16})
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    sbctest()
    # dptest()
