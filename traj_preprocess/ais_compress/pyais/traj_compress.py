import pandas as pd
import numpy as np
from vincenty import vincenty
from math import sqrt
from functools import partial
import sys


def sbc(s, max_dist_threhold: float, max_spd_threhold: float):
    '''
    :param s: raw trajectory
    :param max_dist_threhold: distance threshold
    :param max_spd_threhold: speed threshold
    :return: compressed trajectory
    '''
    scopy = pd.DataFrame.copy(s, deep=True)
    s = scopy.reset_index(drop=True)
    if len(s) <= 2:
        return s
    else:
        is_halt = False
        e = 1
        while e < len(s) and not is_halt:
            i = 1
            while i < e and not is_halt:
                deltae = s.at[e, 'DRGPSTIME'] - s.at[0, 'DRGPSTIME']
                deltai = s.at[i, 'DRGPSTIME'] - s.at[0, 'DRGPSTIME']
                xp = s.at[0, 'DRLATITUDE'] + (s.at[e, 'DRLATITUDE'] - s.at[0, 'DRLATITUDE']) * deltai / deltae
                yp = s.at[0, 'DRLONGITUDE'] + (s.at[e, 'DRLONGITUDE'] - s.at[0, 'DRLONGITUDE']) * deltai / deltae
                ptp = (xp, yp)
                pta = (s.at[i, 'DRLATITUDE'], s.at[i, 'DRLONGITUDE'])
                Vi_1 = vincenty(pta, (s.at[i - 1, 'DRLATITUDE'], s.at[i - 1, 'DRLONGITUDE'])) / (
                        s.at[i, 'DRGPSTIME'] - s.at[i - 1, 'DRGPSTIME']) * 1000 / 1852.25 * 3600
                Vi = vincenty((s.at[i + 1, 'DRLATITUDE'], s.at[i + 1, 'DRLONGITUDE']), pta) / (
                        s.at[i + 1, 'DRGPSTIME'] - s.at[i, 'DRGPSTIME']) * 1000 / 1852.25 * 3600
                if vincenty(pta, ptp) * 1000 / 1852.25 > max_dist_threhold or abs(Vi_1 - Vi) > max_spd_threhold:
                    is_halt = True
                else:
                    i = i + 1

            if is_halt:
                return pd.concat([s[0:1], sbc(s[i:], max_dist_threhold, max_spd_threhold)], ignore_index=True)
            e = e + 1
        if not is_halt:
            return s.loc[[0, len(s) - 1]]


class douglaspeucker:
    def __init__(self, points):
        """
        Rarmer Douglas Peucker
        :param points: trajectory points
        """
        self.points = points

    def point_distance_line(self, pt, point1, point2):
        vec1 = np.array(point1) - np.array(pt)
        vec2 = np.array(point2) - np.array(pt)
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(np.array(point1) - np.array(point2))
        return distance

    def __deviations(self):
        deviations = []
        if len(self.points) > 2:
            for i in range(2, len(self.points)):
                p1 = self.points[i - 2]
                p2 = self.points[i - 1]
                p3 = self.points[i]
                dev = self.point_distance_line(p2, p1, p3)
                deviations.append(dev)
        return deviations

    def avg(self):
        values = self.__deviations()
        if len(values) > 0:
            mean = np.mean(values)
        else:
            mean = 0
        return mean

    def max(self):
        values = self.__deviations()
        mx = np.max(values)
        return mx

    def percent(self):
        values = self.__deviations()
        p = np.percentile(values, 75)
        return p

    def pldist(point, start, end):
        """
        Calculates the distance from ``point`` to the line given
        by the points ``start`` and ``end``.
        :param point: a point
        :type point: numpy array
        :param start: a point of the line
        :type start: numpy array
        :param end: another point of the line
        :type end: numpy array
        """
        if np.all(np.equal(start, end)):
            return np.linalg.norm(point - start)

        return np.divide(
            np.abs(np.linalg.norm(np.cross(end - start, start - point))),
            np.linalg.norm(end - start))

    def rdp_rec(self, M, epsilon, dist=pldist):
        """
        Simplifies a given array of points.
        Recursive version.
        :param M: an array
        :type M: numpy array
        :param epsilon: epsilon in the rdp algorithm
        :type epsilon: float
        :param dist: distance function
        :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
        """
        dmax = 0.0
        index = -1

        for i in range(1, M.shape[0]):
            d = dist(M[i], M[0], M[-1])

            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            r1 = self.rdp_rec(M[:index + 1], epsilon, dist)
            r2 = self.rdp_rec(M[index:], epsilon, dist)

            return np.vstack((r1[:-1], r2))
        else:
            return np.vstack((M[0], M[-1]))

    def _rdp_iter(self, M, start_index, last_index, epsilon, dist=pldist):
        stk = []
        stk.append([start_index, last_index])
        global_start_index = start_index
        indices = np.ones(last_index - start_index + 1, dtype=bool)

        while stk:
            start_index, last_index = stk.pop()

            dmax = 0.0
            index = start_index

            for i in range(index + 1, last_index):
                if indices[i - global_start_index]:
                    d = dist(M[i], M[start_index], M[last_index])
                    if d > dmax:
                        index = i
                        dmax = d

            if dmax > epsilon:
                stk.append([start_index, index])
                stk.append([index, last_index])
            else:
                for i in range(start_index + 1, last_index):
                    indices[i - global_start_index] = False

        return indices

    def rdp_iter(self, M, epsilon, dist=pldist, return_mask=False):
        """
        Simplifies a given array of points.
        Iterative version.
        :param M: an array
        :type M: numpy array
        :param epsilon: epsilon in the rdp algorithm
        :type epsilon: float
        :param dist: distance function
        :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
        :param return_mask: return the mask of points to keep instead
        :type return_mask: bool
        """
        mask = self._rdp_iter(M, 0, len(M) - 1, epsilon, dist)

        if return_mask:
            return mask

        return M[mask]

    def rdp(self, M, epsilon=0, dist=pldist, algo="iter", return_mask=False):
        """
        Simplifies a given array of points using the Ramer-Douglas-Peucker
        algorithm.
        Example:
        >>> from rdp import rdp
        >>> rdp([[1, 1], [2, 2], [3, 3], [4, 4]])
        [[1, 1], [4, 4]]
        This is a convenience wrapper around both :func:`rdp.rdp_iter`
        and :func:`rdp.rdp_rec` that detects if the input is a numpy array
        in order to adapt the output accordingly. This means that
        when it is called using a Python list as argument, a Python
        list is returned, and in case of an invocation using a numpy
        array, a NumPy array is returned.
        The parameter ``return_mask=True`` can be used in conjunction
        with ``algo="iter"`` to return only the mask of points to keep. Example:
        >>> from rdp import rdp
        >>> import numpy as np
        >>> arr = np.array([1, 1, 2, 2, 3, 3, 4, 4]).reshape(4, 2)
        >>> arr
        array([[1, 1],
               [2, 2],
               [3, 3],
               [4, 4]])
        >>> mask = rdp(arr, algo="iter", return_mask=True)
        >>> mask
        array([ True, False, False,  True], dtype=bool)
        >>> arr[mask]
        array([[1, 1],
               [4, 4]])
        :param M: a series of points
        :type M: numpy array with shape ``(n,d)`` where ``n`` is the number of points and ``d`` their dimension
        :param epsilon: epsilon in the rdp algorithm
        :type epsilon: float
        :param dist: distance function
        :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
        :param algo: either ``iter`` for an iterative algorithm or ``rec`` for a recursive algorithm
        :type algo: string
        :param return_mask: return mask instead of simplified array
        :type return_mask: bool
        """

        if algo == "iter":
            algo = partial(self.rdp_iter, return_mask=return_mask)
        elif algo == "rec":
            if return_mask:
                raise NotImplementedError("return_mask=True not supported with algo=\"rec\"")
            algo = self.rdp_rec

        if "numpy" in str(type(M)):
            return algo(M, epsilon, dist)

        return algo(np.array(M), epsilon, dist).tolist()
