import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyais import geotostr
from pyais import traj_stay


# Function to know if we have a CCW turn
def CCW(p1, p2, p3):
    if (p3[1] - p1[1]) * (p2[0] - p1[0]) >= (p2[1] - p1[1]) * (p3[0] - p1[0]):
        return True
    return False


def GiftWrapping(S):
    '''
    outline of polygon
    :param S: collection of points
    :return: outline of polygon
    '''
    print(plt.style.available)  # 查看可用风格
    index = 0
    n = len(S)
    P = [None] * n
    l = np.where(S[:, 0] == np.min(S[:, 0]))
    pointOnHull = S[l[0][0]]
    i = 0
    while True:
        P[i] = pointOnHull
        endpoint = S[0]
        for j in range(1, n):
            if (endpoint[0] == pointOnHull[0] and endpoint[1] == pointOnHull[1]) or not CCW(S[j], P[i], endpoint):
                endpoint = S[j]
        i = i + 1
        pointOnHull = endpoint
        index += 1
        if endpoint[0] == P[0][0] and endpoint[1] == P[0][1]:
            break
    for i in range(n):
        if P[-1] is None:
            del P[-1]
    P = np.array(P)
    # Plot final hull
    centriod = polygoncenterofmass(P.tolist())
    print(geotostr.longgeotostr(centriod[0]))
    print(geotostr.latgeotostr(centriod[1]))
    plt.clf()
    plt.plot(P[:, 0], P[:, 1], 'b-', picker=5)
    plt.plot([P[-1, 0], P[0, 0]], [P[-1, 1], P[0, 1]], 'b-', picker=5, label='The boundary of anchor berth')
    plt.plot(S[:, 0], S[:, 1], ".g", label='The anchor points')
    plt.plot(centriod[0], centriod[1], "or", label='The approximate location of the anchor')
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title('Calculation of Anchor Berth Area', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)
    plt.pause(1)
    return P


def shoelace(list):
    '''
    shoelace formula, calculate area
    :param list: input in clockwise direction
    :return: area of polygon
    '''
    list.append(list[0])
    n = len(list)
    t1 = 0.0
    t2 = 0.0
    for i in range(n - 1):
        t1 += (list[i][0] * list[i + 1][1]) * 3600  # 平方海里，′
        t2 += (list[i][1] * list[i + 1][0]) * 3600
    s = abs(t1 - t2) * 0.5 * 1852.25 * 1852.25
    print(str(s))
    return s


def polygonarea(pointLists):
    area = 0.0
    for i in range(len(pointLists)):
        j = (i + 1) % len(pointLists)
        area += pointLists[i][1] * pointLists[j][0]
        area -= pointLists[i][0] * pointLists[j][1]
    area /= 2.0
    return (abs(area))


def polygoncenterofmass(pointLists):
    '''
    centroid of polygon
    :param pointLists: boundary points
    :return: centroid of polygon
    '''
    if len(pointLists) < 3:
        return (0, 0)
    else:
        cx = 0
        cy = 0
        factor = 0
        j = 0
        a = polygonarea(pointLists)
        for i in range(len(pointLists)):
            j = (i + 1) % len(pointLists)
            factor = pointLists[i][1] * pointLists[j][0] - pointLists[j][1] * pointLists[i][0]
            cx += (pointLists[i][1] + pointLists[j][1]) * factor
            cy += (pointLists[i][0] + pointLists[j][0]) * factor
        factor = 1.0 / (6.0 * a)
        cx = cx * factor
        cy = cy * factor
        return [abs(cy), abs(cx)]


def main():
    font_legend = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 14
    }
    trajdata = traj_stay.stay_st_detect('./data/1.csv')
    scatterColors = ['blue', 'red', 'yellow', 'cyan', 'purple', 'orange', 'olive', 'brown', 'black', 'm']
    for shipmmsi, dt in trajdata.groupby('DRMMSI'):
        labels = dt['SP_Status'].values
        lbs = set(labels)
        plt.plot(dt['DRLONGITUDE'].values, dt['DRLATITUDE'].values, marker='o',
                 markeredgewidth=1.0, linewidth=0.75, label='Interpolated trajectory',
                 markerfacecolor='white', markeredgecolor='m', ms=4, alpha=1.0, color='m', zorder=2)
        for i, item in enumerate(lbs):
            if item >= 0:
                colorSytle = scatterColors[i % len(scatterColors)]
                subCluster = dt.query("SP_Status==@item")
                plt.scatter(subCluster['DRLONGITUDE'].values, subCluster['DRLATITUDE'].values, marker='*', s=70,
                            edgecolors=colorSytle, label='Stay point',
                            c='white', linewidths=1.0, zorder=3)
                plt.title('Stay Point Identification', fontdict={'family': 'Times New Roman', 'size': 16})
                plt.yticks(fontproperties='Times New Roman', size=12)
                plt.xticks(fontproperties='Times New Roman', size=12)
                plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 14})
                plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 14})
                plt.ticklabel_format(useOffset=False, style='plain')
                plt.legend(loc="best", prop=font_legend)
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                subCluster.rename(columns={'DRLATITUDE': 'latitude', 'DRLONGITUDE': 'longitude'},
                                  inplace=True)
                points = []
                for idx, data in subCluster.iterrows():
                    p = [data['longitude'], data['latitude']]
                    points.append(p)
                P = np.array(points)
                L = GiftWrapping(P)
                mj = shoelace(L.tolist())
                print(str(mj))
            else:
                continue


if __name__ == '__main__':
    main()
