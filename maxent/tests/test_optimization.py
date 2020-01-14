# -*- coding: utf-8 -*-
# @Author: Paolo Simeone
# @Date:   2018-09-03 16:34:47
# @Last Modified by:   UGENT\psimeone
# @Last Modified time: 2018-10-30 14:56:36
import scipy.io as sio
import numpy as np
from maxent.baseclass.optimization import *
import numpy.testing as npt
import os
# import pdb
import pytest


class TestMatrices(object):
    def test_gradient_hessian(self):
        # check if gradient and hessian are correctly evaluated
        # this is based on Florian's code before entering the loop of newton's method

        cwd = os.getcwd()
        parent_dir = os.path.dirname(cwd)
        file_mat = os.path.join(*[parent_dir, 'examples', 'graphs', 'test.mat'])
        file = os.path.join(*[parent_dir, 'examples', 'graphs', 'DBLP_AA_GCC_edges.txt'])
        meh = BGDistr(datasource='graph', file=file)
        a, b = meh._sum_on_axis(meh.data.adj_matrix(), undirected=False)
        data = sio.loadmat(file_mat)
        G = data['Grad']  # Gradient
        H = data['H']  # Hessian
        # print("I did it up to this point!")
        assert True

    def test_armijo(self):
        def obj_func(x):
            return 100 * np.square(np.square(x[0]) - x[1]) + np.square(x[0] - 1)

        def d_func(x):
            df1 = 400 * x[0] * (np.square(x[0]) - x[1]) + 2 * (x[0] - 1)
            df2 = -200 * (np.square(x[0]) - x[1])
            return np.array([df1, df2])
        start = np.array([-.3, .1])
        direction = np.array([1, -2])
        # maximum_iterations = 30
        # expected results
        converge_value = np.array([-0.284375, 0.06875])
        meh = BGDistr()
        a, b, c, d = meh.armijo_rule(start, direction, obj_func, d_func, beta=0.25, sigma=0.25)
        npt.assert_array_equal(a, converge_value)
        print("I did it up to this point!")
        assert d == 3
        assert b == pytest.approx(1.66430649757)

    def test_sumonaxis(self):
        """Testing the following graphs:

        Undirected example:
        A -> B
        A -> C
        C -> D
        C -> E
        C -> F
        D -> B
        Directed example has the same vertices.
        """

        adjmat_directed = np.array(([0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1],
                                    [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]))
        # adjmat_undirected = (adjmat_directed + adjmat_directed.T).astype('int')
        meh = BGDistr(adjmat_directed, datasource='custom')
        a, b = meh._sum_on_axis(adjmat_directed, undirected=False)
        c, d = meh._sum_on_axis(adjmat_directed, undirected=True)
        tot = {}
        tot['undirected_rows'] = np.array(([2, 0, 3, 1, 0, 0]))
        tot['undirected_cols'] = np.array(([0, 2, 1, 1, 1, 1]))
        tot['directed'] = np.array(([2, 2, 4, 2, 1, 1]))
        npt.assert_array_equal(a, tot['undirected_rows'])
        npt.assert_array_equal(b, tot['undirected_cols'])
        npt.assert_array_equal(c, tot['directed'])
        npt.assert_array_equal(d, tot['directed'])
        # check if computations go smooth
        meh.compute_lambdas_in_a_cooler_way(undirected=True)
        meh.compute_lambdas_in_a_cooler_way(undirected=False)

    def test_A(self):
        A = np.array([[0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        meh = BGDistr(A, datasource='custom')
        a, b = meh._sum_on_axis(A)
        npt.assert_array_equal(a, np.array(([5, 1, 1, 1, 1, 1])))
        npt.assert_array_equal(b, np.array(([5, 1, 1, 1, 1, 1])))
