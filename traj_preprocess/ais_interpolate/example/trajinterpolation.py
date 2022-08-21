import pandas as pd
import matplotlib.pyplot as plt
from pyais import traj_interpolation


def main():
    traj_file = '../data/1.csv'
    res = 30
    num = None
    gpx_data = traj_interpolation.traj_load(traj_file)
    traj_data_interp = traj_interpolation.traj_interpolate(traj_file, res, num)

    plt.plot(gpx_data['lon'], gpx_data['lat'], marker='*', ms=5, linestyle='--', color='red', linewidth=0.5,
             label='Raw trajectory')
    plt.plot(traj_data_interp['lon'], traj_data_interp['lat'], marker='o', ms=2, linestyle='-', color='g',
             linewidth=0.5, label='Interpolated trajectory')
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title('Trajectory Interpolation', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
