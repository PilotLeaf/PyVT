import pandas as pd
import math
import numpy as np


def latgeotostr(v):
    lbl = 'N'
    if float(v) >= 0:
        lbl = 'N'
    else:
        lbl = 'S'
    v = float(v)
    d = math.floor(v)
    m = (v - d) * 60
    lat = str(d) + '°' + str(np.round(m, 3)) + '′' + lbl
    return lat


def longgeotostr(v):
    lbl = 'E'
    if float(v) >= 0:
        lbl = 'E'
    else:
        lbl = 'W'
    v = float(v)
    d = math.floor(v)
    m = (v - d) * 60
    longi = str(d) + '°' + str(np.round(m, 3)) + '′' + lbl
    return longi


if __name__ == '__main__':
    L = 336
    s = 3.1415926 * (3 * 25 + 90 + L) * (3 * 25 + 90 + L)
    print(str(s))
