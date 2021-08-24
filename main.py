# References:
# [1] https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
# [2] https://www.geeksforgeeks.org/given-a-set-of-line-segments-find-if-any-two-segments-intersect/
# [3] https://www.cgmi.uni-konstanz.de

import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import collections as mc
from sortedcontainers import SortedList, SortedDict, SortedSet
from math import isclose

random.seed(89)

# Reference: [3]
class LineSegment:
    epsilon = 0.000000001

    def __init__(self, x1, y1, x2, y2):
        self.pointA = np.array([x1, y1])
        self.pointB = np.array([x2, y2])

    def intersects(self, other_segment):
        # Idea: We have two lines given by the equation
        # L = p + tv where p is the support point and v the direction vector
        # We can set L_1 = L_2 and obtain a linear equation system which
        # we can solve with Cramer's rule (Ax = b -> x_i = det(A_i)/det(A))
        #
        # The equation system looks like this:
        #
        # x1 + t * (x2 - x1) = x3 + u * (x4 - x3)
        # y1 + t * (y2 - y1) = y3 + u * (y4 - y3)
        #
        # Which we can reshape to
        #
        # [(x2 - x1)  (x3 - x4)] * [ t ] = [(x3 - x1)]
        # [(y2 - y1)  (y3 - y4)]   [ u ]   [(y3 - y1)]
        #
        #            A               x          b
        #
        # Note: you can reshape in at least two ways here, so your solution might
        # deviate from sth in the book or online but still be equivalent!

        x1, y1 = self.pointA
        x2, y2 = self.pointB
        x3, y3 = other_segment.pointA
        x4, y4 = other_segment.pointB

        determinant = ((x2 - x1) * (y3 - y4) - (y2 - y1) * (x3 - x4))
        # print("Det:", determinant)

        # handle colinear
        if abs(determinant) < LineSegment.epsilon:
            return 0, 0, None

        # t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / determinant
        t = ((x3 - x1) * (y3 - y4) - (y3 - y1) * (x3 - x4)) / determinant
        u = ((x2 - x1) * (y3 - y4) - (y1 - y3) * (x2 - x4)) / determinant

        pX = x1 + t * (x2 - x1)
        pY = y1 + t * (y2 - y1)

        # print(t, u)
        intersects = True if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else False

        # Does this already cover all cases?

        return pX, pY, intersects

    def plot(self, ax):
        ax.plot([self.pointA[0], self.pointB[0]],
                [self.pointA[1], self.pointB[1]])


def find_intersection_l_segments(x1, y1, x2, y2, x3, y3, x4, y4):
    if max(x1, x2) < min(x3, x4):
        return -1, -1, False

    da1 = x1 - x2
    da2 = x3 - x4

    if da1 != 0 and da2 != 0:
        a1 = (y1 - y2) / da1
        a2 = (y3 - y4) / da2
    else:
        return -1, -1, False

    b1 = y1 - a1 * x1
    b2 = y3 - a2 * x3

    # Check if parallel
    if a1 == a2:
        return -11, -11, False

    d3 = a1 - a2
    if d3 != 0:
        xa = (b2 - b1) / d3
    else:
        return -111, -111, False

    ya = a1 * xa + b1

    v1 = min(x1, x2)
    v2 = min(x3, x4)
    v3 = max(v1, v2)
    v4 = max(x1, x2)
    v5 = max(x3, x4)
    v5 = min(v4, v5)

    if ((xa + 0.00003 < v3) or
            (xa - 0.00003 > v5)):
        # print("Out of bound intersection. " + "xa: " + str(xa) + " v3: " + str(v3) + " v5: " + str(v5))

        return -1111, -1111, False
    else:
        # print("Intersection point:" + str(xa) + ", ", str(ya))
        return xa, ya, True


def create_lines(amount_of_lines_to_create):
    # Creating random line points x_1,y_1,x_2,y_2.
    # sl = SortedList()
    sl = np.zeros(shape=(amount_of_lines_to_create + 1, 2, 2))
    mq = SortedList()

    # for i in range(amount_of_lines_to_create):

    for i in range(amount_of_lines_to_create):
        randx_1 = random.uniform(0, 1) * 100.0
        randy_1 = random.uniform(0, 1) * 100.0
        randx_2 = random.uniform(0, 1) * 100.0
        randy_2 = random.uniform(0, 1) * 100.0

        # Adjusting the lengths of the lines to min:20.0
        if abs(randx_2 - randx_1) <= 20.0:
            randx_1 += 20.0
        if abs(randy_2 - randy_1) <= 20.0:
            randy_1 += 20.0

        if randy_1 < randy_2:
            randy_1, randy_2 = randy_2, randy_1
        sl[i] = ([[randx_1, randy_1], [randx_2, randy_2]])
        mq.add([randy_1, randx_1])
        mq.add([randy_2, randx_2])
        print("Created random y and x points: " + str(sl[i]))

    # Plotting the created lines.
    # print(np.shape(np_array))

    lc = mc.LineCollection(sl, color="cornflowerblue", linewidths=1)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_title('Created Lines')

    fig.show()

    # Returns created lines array.
    return sl, mq


def sweeping_line(np_lines, e_point, s_line, mq, active_segments, found_intersections):
    val = input("Please enter an input(n: Sweep line moves to next event. e: Exit): ")

    the_range = (len(np_lines) - 1) * 2
    print("e_point: " + str(e_point) + " mq len: " + str(len(mq)))
    if e_point < len(mq):
        if val == 'e':
            finish_line = True
        elif val == 'n':
            # print("Previous sweep line: " + str(s_line))
            new_y = len(mq) - e_point - 1
            s_line[0][1] = mq[new_y][0]
            s_line[1][1] = mq[new_y][0]
            # print("New sweep line: " + str(s_line))
            len_arr = len(np_lines)
            # print(len_arr-1)
            for k in range(len_arr):
                if np_lines[k][1][1] <= s_line[1][1]:
                    # print(k)

                    ix, iy, res = find_intersection_l_segments(np_lines[k][0][0], np_lines[k][0][1],
                                                               np_lines[k][1][0], np_lines[k][1][1],
                                                               s_line[0][0], s_line[0][1],
                                                               s_line[1][0], s_line[1][1])

                    if res:
                        active_segments.add([[ix], [np_lines[k][0][0], np_lines[k][0][1],
                                                    np_lines[k][1][0], np_lines[k][1][1]]])

                        # which_line_segment_index = active_segments.bisect_left([ix])
                        print(ix)
                        print(active_segments)
                        t_v = [[ix], [np_lines[k][0][0], np_lines[k][0][1],
                                      np_lines[k][1][0], np_lines[k][1][1]]]
                        wlsi = active_segments.index(t_v)
                        print("which_line_segment_index")
                        print(wlsi)
                        if wlsi > 0:

                            ix, iy, res2 = find_intersection_l_segments(active_segments[wlsi][1][0],
                                                                        active_segments[wlsi][1][1],
                                                                        active_segments[wlsi][1][2],
                                                                        active_segments[wlsi][1][3],
                                                                        active_segments[wlsi - 1][1][0],
                                                                        active_segments[wlsi - 1][1][1],
                                                                        active_segments[wlsi - 1][1][2],
                                                                        active_segments[wlsi - 1][1][3])
                            # print(ix, iy, res)
                            if res2:
                                print("FOUND")
                                t1 = [ix, iy]
                                how_many_1 = found_intersections.count(t1)
                                if how_many_1 < 1:
                                    found_intersections.add([ix, iy])
                                    print(found_intersections)
                                t = [iy, -10]
                                print("COUNT IY " + str(mq.count(t)))
                                how_many = mq.count(t)
                                if how_many < 1:
                                    mq.add([iy, -10])

                        if wlsi + 1 < len(active_segments):

                            ix, iy, res2 = find_intersection_l_segments(active_segments[wlsi][1][0],
                                                                        active_segments[wlsi][1][1],
                                                                        active_segments[wlsi][1][2],
                                                                        active_segments[wlsi][1][3],
                                                                        active_segments[wlsi + 1][1][0],
                                                                        active_segments[wlsi + 1][1][1],
                                                                        active_segments[wlsi + 1][1][2],
                                                                        active_segments[wlsi + 1][1][3])
                            # print(ix, iy, res)
                            if res2:
                                print("FOUND")
                                t1 = [ix, iy]
                                how_many_1 = found_intersections.count(t1)
                                if how_many_1 < 1:
                                    found_intersections.add([ix, iy])
                                    print(found_intersections)
                                t = [iy, -10]
                                print("COUNT IY " + str(mq.count(t)))
                                how_many = mq.count(t)
                                if how_many < 1:
                                    mq.add([iy, -10])

            plotlist = np_lines
            plotlist[len(plotlist) - 1] = s_line
            lc = mc.LineCollection(plotlist, color="cornflowerblue", linewidths=1)

            fig, ax = pl.subplots()
            ax.add_collection(lc)
            ax.autoscale()
            ax.margins(0.1)
            ax.set_title('Created Lines')
            f_r = len(found_intersections)
            if f_r > 0:

                c_np = np.zeros([f_r, 2])
                print(c_np)
                print(f_r)
                print(np.size(c_np))
                for ic in range(f_r):
                    c_np[ic][0] = found_intersections[ic][0]
                    c_np[ic][1] = found_intersections[ic][1]
                print(c_np)
                xs = c_np[:, 0]
                print(xs)
                ys = c_np[:, 1]
                print(ys)

                ax.scatter(x=xs, y=ys, color='red')

            fig.show()

            # np_lines[len(np_lines) - 1] = s_line
            print("Active segments: ")
            print(active_segments)
            active_segments = SortedList()

            return sweeping_line(np_lines, e_point + 1, s_line, mq, active_segments, found_intersections)


        else:
            print("You entered a wrong value.")
            sweeping_line(np_lines, e_point, s_line, mq, active_segments, found_intersections)

    else:
        print("Found intersections:")
        print(found_intersections)

        return found_intersections


if __name__ == "__main__":
    fig, ax = plt.subplots()
    # l1 = LineSegment(0, 0, 1, 0)
    # l2 = LineSegment(0.5, 0, 0.5, 1)

    l1 = LineSegment(0, 0, 1, 1)
    l2 = LineSegment(0, 1, 1, 0)
    s_line = [[-10, 130], [130, 130]]
    active_segments = SortedList()
    found_intersections = SortedList()
    found_intersections_r = SortedList()
    l1.plot(ax)
    l2.plot(ax)

    # x, y, i = l2.intersects(l1)
    x, y, i = l1.intersects(l2)

    if i is None:
        print("None")
        pass
    elif i is True:
        print("True")
        plt.scatter(x, y, color='g')
    else:
        print("False")
        plt.scatter(x, y, color='r')

    np_lines, mq = create_lines(6)
    found_intersections_r = sweeping_line(np_lines, 0, s_line, mq, active_segments, found_intersections)

    print("Returned result: ")
    print(found_intersections_r)

    # p = sort_list_try()
    # print(p)
    # plt.show()
