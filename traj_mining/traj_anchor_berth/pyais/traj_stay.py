import numpy as np
import pandas as pd
import copy
import time
import math
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from pyais import traj_clean
from pyais import traj_interpolation
from pyais import traj_segment

import warnings

warnings.filterwarnings("ignore")


# Part1

def compute_squared_EDM(X):
    return squareform(pdist(X, metric='euclidean'))


def ST_DBSCAN(data, eps1: float, eps2: float, minPts: int):
    '''
    :param data: interpolated trajectory
    :param eps1: spatial neighborhood
    :param eps2: time neighborhood
    :param minPts: the minimum number of points that satisfy the double-neighborhood
    :return: labels
    '''
    n, m = data.shape
    timeDisMat = compute_squared_EDM(data[:, 0].reshape(n, 1))
    disMat = compute_squared_EDM(data[:, 1:])
    core_points_index = np.where(np.sum(np.where((disMat <= eps1) & (timeDisMat <= eps2), 1, 0), axis=1) >= minPts)[0]
    labels = np.full((n,), -1)
    clusterId = 0
    for pointId in core_points_index:
        if (labels[pointId] == -1):
            labels[pointId] = clusterId
            neighbour = np.where((disMat[:, pointId] <= eps1) & (timeDisMat[:, pointId] <= eps2) & (labels == -1))[0]
            seeds = set(neighbour)
            while len(seeds) > 0:
                newPoint = seeds.pop()
                labels[newPoint] = clusterId
                queryResults = set(np.where((disMat[:, newPoint] <= eps1) & (timeDisMat[:, newPoint] <= eps2))[0])
                if len(queryResults) >= minPts:
                    for resultPoint in queryResults:
                        if labels[resultPoint] == -1:
                            seeds.add(resultPoint)
            clusterId = clusterId + 1
    return labels


def stay_st_detect(traj_file: str):
    '''
    :param traj_data: raw trajectory filename
    :return: interpolated trajectory,labels
    '''
    trajdf = pd.read_csv(traj_file, encoding='gbk')
    segdf = traj_segment.segment(trajdf, 1500)
    cleandf = traj_clean.heuristic_clean(segdf, 25)

    tdf = pd.DataFrame(columns=cleandf.columns, index=cleandf.index)
    tdf.drop(tdf.index, inplace=True)
    tdf['SP_Status'] = 0

    for shipmmsi, dt in cleandf.groupby('DRMMSI'):
        data = dt.copy(deep=True)
        data.sort_values(by='DRGPSTIME', ascending=True, inplace=True)
        data = data.reset_index(drop=True)
        trajs_data = {'lat': [], 'lon': [], 'tstamp': [], 'speed': []}
        for index, row in data.iterrows():
            trajs_data['lat'].append(row['DRLATITUDE'])
            trajs_data['lon'].append(row['DRLONGITUDE'])
            trajs_data['speed'].append(row['DRSPEED'])
            trajs_data['tstamp'].append(row['DRGPSTIME'])
        if len(trajs_data['lon']) < 4:
            continue
        res = 30.0
        my_traj_data_interp = traj_interpolation.traj_interpolate_df(trajs_data, res, None)
        dataDf = pd.DataFrame(my_traj_data_interp)
        dataDf.columns = ['lat', 'lon', 'spd', 'ts']
        dataDf['ts'] = dataDf['ts'].map(lambda x: int(x))
        datanew = dataDf[['ts', 'lat', 'lon']].values

        minpts = round(np.log(len(datanew)))
        if minpts < 4:
            minpts = 4
        ep2 = minpts / 2 * res
        ep1 = 2.2 * ep2 / 3600.0 / 60.0
        tj = pd.DataFrame(datanew)
        tj.columns = ['DRGPSTIME', 'DRLATITUDE', 'DRLONGITUDE']
        labels = ST_DBSCAN(datanew, ep1, ep2, minpts)
        tj['SP_Status'] = labels
        tj['DRMMSI'] = shipmmsi
        tj['DRGPSTIME'] = tj['DRGPSTIME'].map(lambda x: int(x))
        tdf = tdf.append(tj, ignore_index=True)
        tdf = tdf.fillna(0)
    return tdf


# Part2
def rad(d):
    return float(d) * math.pi / 180.0


EARTH_RADIUS = 6378.137


def GetDistance(lng1, lat1, lng2, lat2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    s = round(s * 10000, 2) / 10
    return s


def ka(dis, dc):
    if (dis >= dc):
        return 0
    else:
        return 1


def density(data, dc):
    dc = float(dc)
    latitude = list(data['DRLATITUDE'])
    longitude = list(data['DRLONGITUDE'])
    part_density = []  # 存储局部密度值
    scope = []  # 记录每个点计算局部密度的范围
    leftBoundary = 0
    rightBoundary = len(data) - 1  # 左边界与右边界
    for i in range(len(data)):
        trigger = True
        left = i - 1
        right = i + 1
        incrementLeft = 1
        incrementRight = 1
        while trigger:
            # 向左拓展
            if incrementLeft != 0:
                if left < 0:
                    left = leftBoundary
                distanceLeft = GetDistance(longitude[left], latitude[left], longitude[i], latitude[i])
                if (distanceLeft < dc) & (left > leftBoundary):
                    left -= 1
                else:
                    incrementLeft = 0
            # 向右拓展
            if incrementRight != 0:
                if right > rightBoundary:
                    right = rightBoundary
                distanceRight = GetDistance(longitude[i], latitude[i], longitude[right], latitude[right])
                if (distanceRight < dc) & (right < rightBoundary):
                    right += 1
                else:
                    incrementRight = 0
            # 若左右都停止了拓展，此点的局部密度计算结束
            if (incrementLeft == 0) & (incrementRight == 0):
                trigger = False
            if (left == leftBoundary) & (incrementRight == 0):
                trigger = False
            if (incrementLeft == 0) & (right == rightBoundary):
                trigger = False
        if left == leftBoundary:
            scope.append([left, right - 1])
            part_density.append(right - left - 1)
        elif right == rightBoundary:
            scope.append([left + 1, right])
            part_density.append(right - left - 1)
        else:
            scope.append([left + 1, right - 1])
            part_density.append(right - left - 2)
    return part_density, scope


# 反向更新的方法
def SP_search(data, tc):
    tc = int(tc)
    SP = []
    part_density = copy.deepcopy(list(data['part_density']))
    scope = copy.deepcopy(list(data['scope']))
    latitude = copy.deepcopy(list(data['DRLATITUDE']))
    longitude = copy.deepcopy(list(data['DRLONGITUDE']))
    timestamp = copy.deepcopy(list(data['DRGPSTIME']))
    trigger = True
    used = []
    while trigger:
        partD = max(part_density)
        index = part_density.index(partD)
        print('index:', index)
        start = scope[index][0]
        end = scope[index][1]
        if len(used) != 0:
            for i in used:
                if (scope[i][0] > start) & (scope[i][0] < end):
                    part_density[index] = scope[i][0] - start - 1
                    scope[index][1] = scope[i][0] - 1
                    print("1_1")
                if (scope[i][1] > start) & (scope[i][1] < end):
                    part_density[index] = end - scope[i][1] - 1
                    scope[index][0] = scope[i][1] + 1
                    print("1_2")
                if (scope[i][0] <= start) & (scope[i][1] >= end):
                    part_density[index] = 0
                    scope[index][0] = 0;
                    scope[index][1] = 0
                    print("1_3")
            start = scope[index][0];
            end = scope[index][1]
        timeCross = timestamp[end] - timestamp[start]
        print('time:', timeCross)
        if timeCross > tc:
            SarvT = time.localtime(timestamp[start])
            SlevT = time.localtime(timestamp[end])
            SP.append([index, latitude[index], longitude[index], SarvT, SlevT, scope[index]])
            used.append(index)
            for k in range(scope[index][0], scope[index][1] + 1):
                part_density[k] = 0
        part_density[index] = 0
        if max(part_density) == 0:
            trigger = False
    return SP


# 通过代表的距离是否小于距离阈值来大致判断停留点是否一致
def similar(sp, dc):
    dc = float(dc)
    latitude = copy.deepcopy(list(sp['latitude']))
    longitude = copy.deepcopy(list(sp['longitude']))
    i = 0;
    index = list(sp.index)
    for i in index:
        for j in index:
            if i != j:
                dist = GetDistance(longitude[i], latitude[i], longitude[j], latitude[j])
                if dist < 1.5 * dc:
                    sp = sp.drop(j, axis=0)
                    index.remove(j)
    return sp


def stay_spt_detect(traj_file: str):
    '''
    :param traj_data: raw trajectory filename
    :return: interpolated trajectory,labels
    '''
    trajdf = pd.read_csv(traj_file, encoding='gbk')
    segdf = traj_segment.segment(trajdf, 1500)
    cleandf = traj_clean.heuristic_clean(segdf, 25)

    tdf = pd.DataFrame(columns=cleandf.columns, index=cleandf.index)
    tdf.drop(tdf.index, inplace=True)
    tdf['SP_Status'] = 0

    for shipmmsi, dt in cleandf.groupby('DRMMSI'):
        if len(str(shipmmsi)) > 5:
            if len(dt) > 10:
                data = dt.copy(deep=True)
                data = data.reset_index(drop=True)
                sj = data.copy(deep=True)
                sj['SP_Status'] = 0

                dc = 400
                tc = 600
                data['part_density'], data['scope'] = density(data, dc)
                SP = SP_search(data, tc)
                output = pd.DataFrame(SP)

                if output.empty == False:
                    output.columns = ['index', 'longitude', 'latitude', 'arriveTime', 'leftTime', 'scope']
                    output = similar(output, dc)

                    for sco in output['scope']:
                        start, end = sco[0], sco[1]
                        tcog = sj['DRDIRECTION'][start:end]
                        ss = 0  # 1靠泊、2锚泊
                        if tcog.var() < 800:
                            ss = 1
                        else:
                            ss = 2
                        sj.loc[start:end, 'SP_Status'] = ss
                    tdf = tdf.append(sj, ignore_index=True)
    return tdf
