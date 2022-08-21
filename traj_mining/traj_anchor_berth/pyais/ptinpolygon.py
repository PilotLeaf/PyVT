import numpy as np
import math


def isPointInPoly(point, poly):
    '''Determine whether a point is inside an arbitrary polygon(convex or non-convex)
    Input:
    point: tuple(x, y)
    poly: [x1, y1, x2, y2, ..., xn, yn]
    Output:
    True: inside
    False: outside
    '''
    assert len(poly) % 2 == 0

    vector_list = []
    angle_sum = 0
    poly = np.array(poly)

    for i in range(int(len(poly) / 2)):
        vector = [poly[i << 1] - point[0], poly[(i << 1) + 1] - point[1]]
        vector = np.array(vector)
        vector_list.append(vector)

    for i in range(len(vector_list)):
        if i == len(vector_list) - 1:
            cross = np.cross(vector_list[i], vector_list[0])
            cos = np.dot(vector_list[i], vector_list[0]) / (
                        np.linalg.norm(vector_list[i]) * np.linalg.norm(vector_list[0]))
        else:
            cross = np.cross(vector_list[i], vector_list[i + 1])
            cos = np.dot(vector_list[i], vector_list[i + 1]) / (
                        np.linalg.norm(vector_list[i]) * np.linalg.norm(vector_list[i + 1]))
        try:
            angle = math.acos(cos)
            if cross >= 0:
                angle_sum += angle
            else:
                angle_sum -= angle
        except:
            print(cos)

    if abs(angle_sum) > 6.283185307:
        return True
    else:
        return False

# if __name__ == '__main__':
#     point = (9, 1.1)
#     poly = [1, 1, 1, 10, 10, 10, 5, 5, 10, 1]
#     print(isPointInPoly(point, poly))
