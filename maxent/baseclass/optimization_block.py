from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import networkx.exception as nxexp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from .MRDbase import *


def dispatch(class_u_want):
    """Varies the data container according to the origin of the data"""
    return {
        'graph': lambda: graphdata(),
        'MRD': lambda: MRD(),
        # 'custom': lambda: # matrix submitted by user
    }.get(class_u_want, lambda: None)()


def dispatch_dg(distribution_u_want, a):
    """Varies the expression of gradient according to the distribution type"""
    return {
        'bernoulli': lambda: a / (1. + a),
        'geometric': lambda: a / (1. - a),  # initialization can give divide by zero
        'exponential': lambda: a,
    }.get(distribution_u_want, lambda: None)()


def dispatch_dh(distribution_u_want, a):
    """Varies the expression of hessian according to the distribution type"""
    return {
        'bernoulli': lambda: a / ((1. + a) ** 2),
        'geometric': lambda: a / ((1. - a) ** 2),
        'exponential': lambda: 1. / (a ** 2),
    }.get(distribution_u_want, lambda: None)()


def dispatch_dl(distribution_u_want, a):
    """Varies the expression of lagrangian according to the distribution type"""
    return {
        'bernoulli': lambda: 1. + a,
        'geometric': lambda: 1. / (1. - a),
        'exponential': lambda: -1. / a,
    }.get(distribution_u_want, lambda: None)()


def dispatch_d_1s(distribution_u_want, a):
    """Varies the expression of bg distribution according to the distribution type"""
    return {
        'bernoulli': lambda: 1 / (1. + a),
        'geometric': lambda: 1. - a,
        'exponential': lambda: -a,
    }.get(distribution_u_want, lambda: None)()


class BGDistr(object):

    def __init__(self, *args, **kwargs):
        """Class initialization

        Examples
        --------

        >>> from baseclass.MRDbase import **
        >>> import os
        >>> cwd = os.getcwd()
        >>> file = os.path.join(cwd, *['examples','graphs', 'DBLP_AA_GCC_edges.txt'])
        >>> fileMRD = os.path.join(cwd, *['examples','movies', 'relationships', 'role_in_film.mrd'])
        >>> bd_graph = BGDistr(datasource='graph', file=file)
        >>> bd_MRD = BGDistr(datasource='MRD', file=fileMRD)

        Notes
        -----
        Ulterior things to explain to user.
        """
        super(BGDistr, self).__init__()
        # np.seterr(all='raise')
        try:
            self.datasource = kwargs['datasource']
            self.data = dispatch(self.datasource)
            if kwargs['datasource'] == 'custom':  # this should go into the dispatch
                # maybe it should be scipy.sparse.csr_matrix(args[0])
                self.data = sp.csr_matrix(args[0])
            else:
                self.data.from_file(kwargs['file'], header=True)
        except KeyError:
            print("No datasource or file provided.")
        except TypeError:
            print("The type is not recognized. Choose graph or MRD during initialization.")
        except nxexp.NetworkXException:
            print("Trying to load from a corrupted file or a wrong datasource")
        # else:
        #     # what happens when no kwargs are passed should be defined.
        #     pass

    # Properties

    # Read only Properties

    # Methods

    def _index_uniques(self, M):
        """Short description of function

        Parameters
        ----------
        M : type

            1D matrix

        Returns
        -------
        beans : ndarray

            The sorted unique values.

        index : ndarray

            The indices to reconstruct the original array from the unique array.

        colbeans_frequencies : ndarray

            The number of times each of the unique values comes up in the original array.

        See Also
        --------
        _compute_lambdas, compute_lambdas_in_a_cooler_way

        Examples
        --------
        Write some usage example
        >>> Class.method()
        >>> Class.woof()

        Notes
        -----
        Make it a (read only) property?
        Built some test for dimensionality?
        """

        return np.unique(M, axis=1, return_counts=True, return_inverse=True)

    def _compute_lambdas(self, rowvec, colvec, A, iterations=100, epsilon=1e-10,
                         tol=1e-5, verbose=False, return_prob=False, label_lists=None,
                         distribution_type='bernoulli', method='L-BFGS-B', **kwargs):

        def dispatch_block(x, label_idx_map, n_unique, n_nodes):
            result = np.zeros((n_nodes, n_nodes))
            for i in range(n_unique):
                rids = label_idx_map[i].T
                for j in range(n_unique):
                    result[rids, label_idx_map[j]] = x[i*n_unique+j]
            return result

        def dispatch_block_sum(yo, label_idx_map, n_unique):
            result = np.zeros(n_unique**2)
            for i in range(n_unique):
                rids = label_idx_map[i].T
                for j in range(n_unique):
                    result[i*n_unique + j] = yo[rids, label_idx_map[j]].sum()
            return result

        def dispatch_sum(distribution_u_want, x):
            E = x[:mtilde, None] + x[mtilde:(mtilde+ntilde)]
            ptr = mtilde+ntilde
            for i, btilde in enumerate(btildes):
                E += dispatch_block(x[ptr:ptr+btilde], label_idx_maps[i],
                                    label_n_uniques[i], mtilde)
                ptr += btilde
            return np.exp(E)

        def gradientexpress(x, mtilde, ntilde, btildes, rowbeans, colbeans,
                            blockbeans):
            yo = dispatch_sum(distribution_type, x)
            yo = dispatch_dg(distribution_type, yo)
            yo -= np.diag(np.diag(yo))
            grad = []
            grad.append(yo.sum(axis=1) - rowbeans)
            grad.append(yo.sum(axis=0) - colbeans)
            for i,blockbean in enumerate(blockbeans):
                grad.append(dispatch_block_sum(yo, label_idx_maps[i], label_n_uniques[i]) - blockbean)
            return np.hstack(grad)

        def lagrangian(x, mtilde, ntilde, btildes, rowbeans, colbeans,
                       blockbeans):
            yo = dispatch_sum(distribution_type, x)
            yo = dispatch_dl(distribution_type, yo)
            yo = np.log(yo)
            yo -= np.diag(np.diag(yo))
            yo = np.sum(yo) - np.sum(x[:mtilde] * rowbeans)  \
                 - np.sum(x[mtilde:(mtilde+ntilde)] * colbeans)
            ptr = mtilde + ntilde
            for i,btilde in enumerate(btildes):
                 yo -= np.sum(x[ptr:ptr+btilde]*blockbeans[i])
                 ptr += btilde
            return yo

        label_idx_maps = []
        label_n_uniques = []
        for labels in label_lists:
            label_unique = np.unique(labels)
            label_n_uniques.append(len(label_unique))
            label_idx_maps.append(
                                  [np.where(labels == i)[0][None,:] for i in label_unique]
            )
        colbeans = np.array(colvec).squeeze()
        rowbeans = np.array(rowvec).squeeze()
        blockbeans = []
        for i,label_idx_map in enumerate(label_idx_maps):
            blockbeans.append(dispatch_block_sum(A, label_idx_map, label_n_uniques[i]))

        mtilde = len(rowbeans)  # number of distinct row sums
        ntilde = len(colbeans)  # number of distinct column sums
        btildes = [int(n_unique**2) for n_unique in label_n_uniques]
        btilde = sum(btildes)

        if method == 'L-BFGS-B':
            res = minimize(lagrangian, np.random.randn(mtilde+ntilde+btilde, 1),
                           args=(mtilde, ntilde, btildes, rowbeans, colbeans, blockbeans),
                           method='L-BFGS-B',
                           jac=gradientexpress,
                           options={'disp': True, 'maxcor': 20, 'ftol': 1e-23,
                           'gtol':tol, 'maxiter':1000, 'maxls':20})
            la = res.x
        else:
            raise ValueError('optimization method {:s} doesnot exist.'.format(method))
        E = la[:mtilde, None] + la[mtilde:(mtilde+ntilde)]
        ptr = mtilde+ntilde
        for i, btilde in enumerate(btildes):
            E += dispatch_block(la[ptr:ptr+btilde], label_idx_maps[i],
                                label_n_uniques[i], mtilde)
            ptr += btilde
        return E



    def _sum_on_axis(self, M, undirected=True):
        """Private method to evaluate the sums over the axis

        Parameters
        ----------
        undirected : boolean

            Just specifiying if the graph data are undirected

            (need to find a common interface for MRD and graphs)

        Returns
        -------
        rowsum : ndarray

            numpy array with the cumulative sum over rows

        colsum : ndarray

            numpy array with the cumulative sum over columns

        See Also
        --------
        compute_lambdas_in_a_cooler_way

        Notes
        -----
        Ulterior things to explain to user.
        """

        if undirected:
            M = (M + M.T).astype('bool')
        colsum = M.sum(axis=0)  # , dtype=np.int64)
        rowsum = M.sum(axis=1)  # , dtype=np.int64) # already np.int64
        return rowsum.T, colsum

    def _process_matrices(self, **kwargs):
        """Private method which just serves to wrap/prepare the data before feeding them to the optimization.

        Notes
        -----
        It may be refactored in a smarter way in the future.
        """

        if self.datasource == 'graph':
            # Must store the adj_matrix
            self.data.adjacencymat = self.data.adj_matrix()
            return self._sum_on_axis(self.data.adjacencymat, **kwargs)
        elif self.datasource == 'custom':
            return self._sum_on_axis(self.data, **kwargs)
        elif self.datasource == 'MRD':
            return {k: self._sum_on_axis(self.data.data[k].todense(), undirected=False) for k in self.data.data.keys() if k != 'pk'}

    def compute_lambdas_in_a_cooler_way(self, **kwargs):
        """This function it's juts an interface to distribute data to the optimization algorhitm

        Parameters
        ----------

        undirected : boolean

            If set to True with graph data and custom data, makes the adjacency matrix undirected.

        kwargs :

            Look at the parameters of _compute_lambdas

        Returns
        -------

            See return of _compute_lambdas. If the original data is from a MRD then it's stored in a dictionary.

        See Also
        --------
        _compute_lambdas, _process_matrices

        Examples
        --------
        >>> from baseclass.optimization import *
        >>> import os
        >>> cwd = os.getcwd()
        >>> file = os.path.join(cwd, *['examples','graphs', 'DBLP_AA_GCC_edges.txt'])
        >>> fileMRD = os.path.join(cwd, *['examples','movies', 'relationships', 'role_in_film.mrd'])
        >>> bd_graph = BGDistr(datasource='graph', file=file)
        >>> bd_MRD = BGDistr(datasource='MRD', file=fileMRD)
        >>> lol.compute_lambdas_in_a_cooler_way()
        >>> lol.compute_lambdas_in_a_cooler_way(iterations=1000) # Increase number of default iterations

        Notes
        -----
        Ulterior things to explain to user.
        """

        try:
            undirection = kwargs.pop('undirected')
            if not isinstance(undirection, bool):
                raise ValueError
            print('Undirected value has been set to: ' + str(undirection))
        except KeyError as e:
            print('Undirected value not specified. Using default value: True for graph, False for MRD.')
            undirection = True
        except ValueError as e:
            print('Specified type for Undirected is not bool. Using default value: True for graph, False for MRD.')
            undirection = True
        output = self._process_matrices(undirected=undirection)
        if isinstance(output, dict):
            self.mrd_dict = {k: self._compute_lambdas(*v, **kwargs, current_key=k) for k, v in output.items()}
            return self.mrd_dict
        else:
            return self._compute_lambdas(*output, self.data, **kwargs)
            # return self.lambdas_r, self.lambdas_c, self.lambdas_b




    def get_row_probability(self, row_id, col_ids, distribution_type='bernoulli'):
        '''
        Compute prior (degree) probability for the entries in a row specified
        by row_id.
        '''
        row_la = self.lambdas_r[self.r_indexer[row_id]]
        col_las = self.lambdas_c[self.c_indexer[col_ids]]

        E = np.exp(row_la + col_las)
        P_i = dispatch_d_1s(distribution_type, E) * E


        P_i[list(col_ids).index(row_id)] = 0
        return P_i

    def get_column_probability(self, col_id, row_ids, distribution_type='bernoulli'):
        '''
        Compute prior (degree) probability for the entries in a column
        specified by column.
        '''
        col_la = self.lambdas_c[self.c_indexer[col_id]]
        row_las = self.lambdas_r[self.r_indexer[row_ids]]

        E = np.exp(row_las + col_la)
        P_j = dispatch_d_1s(distribution_type, E) * E

        P_j[list(row_ids).index(col_id)] = 0
        return P_j


    def distribution(self, i, j, **kwargs):
        """ Distribution is just a utility to retrieve the background distribution punctually

        Parameters
        ----------

        i : integer

            Row index.

        j : integer

            Column index

        key : string

            Optional parameter that makes sense only with MRD data.

        Returns
        -------

        value : float

            Value of probability for row i and column j.

        Notes
        -----

            This can be surely coded more elegantly.

        """
        def __get_p_i_j():
            """ Returns the background distribution punctually for graph data


            """
            try:
                lambdas_r = self.lambdas_r
                lambdas_c = self.lambdas_c
                r_indexer = self.r_indexer
                c_indexer = self.c_indexer
                value = self.data.adjacencymat[i, j]
                sum_value = (lambdas_r[r_indexer[i]] + lambdas_c[c_indexer[j]])
            except AttributeError:
                self.lambdas_r, self.lambdas_c, self.r_indexer, self.c_indexer, _, _ = self.compute_lambdas_in_a_cooler_way(**kwargs)
                value = self.data.adjacencymat[i, j]
                sum_value = self.lambdas_r[self.r_indexer[i]] + self.lambdas_c[self.c_indexer[j]]
            except IndexError:
                raise
            return value, sum_value


        def __get_p_i_j_mrd():
            """ Returns the background distribution punctually for MRD data.

            """
            try:
                key = kwargs['key']
                lambdas_r = self.mrd_dict[key][0]
                lambdas_c = self.mrd_dict[key][1]
                r_indexer = self.mrd_dict[key][2]
                c_indexer = self.mrd_dict[key][3]
                value = self.data.data[key][i, j]
                sum_value = (lambdas_r[r_indexer[i]] + lambdas_c[c_indexer[j]])
            except KeyError:
                print('For MRD you need to specify a key.')
                raise
            except AttributeError:
                self.mrd_dict = self.compute_lambdas_in_a_cooler_way(**kwargs)
                value = self.data.data[key][i, j]
                sum_value = self.mrd_dict[key][0][self.mrd_dict[key][2][i]] + self.mrd_dict[key][1][self.mrd_dict[key][3][j]]
            except IndexError:
                raise
            return value, sum_value

        def __get_p_i_j_custom():
            """ Returns the background distribution punctually for custom data.

            """
            try:
                lambdas_r = self.lambdas_r
                lambdas_c = self.lambdas_c
                r_indexer = self.r_indexer
                c_indexer = self.c_indexer
                value = self.data[i, j]
                sum_value = (lambdas_r[r_indexer[i]] + lambdas_c[c_indexer[j]])
            except AttributeError:
                self.lambdas_r, self.lambdas_c, self.r_indexer, self.c_indexer, _, _ = self.compute_lambdas_in_a_cooler_way(**kwargs)
                value = self.data[i, j]
                sum_value = self.lambdas_r[self.r_indexer[i]] + self.lambdas_c[self.c_indexer[j]]
            except IndexError:
                raise
            return value, sum_value

        try:
            distribution_type = kwargs['distribution_type']
        except KeyError:
            print('No type of distribution has been set. Using default value: bernoulli ')
            distribution_type = 'bernoulli'  # default value

        if self.datasource == 'MRD':
            value, sum_value = __get_p_i_j_mrd()
        elif self.datasource == 'graph':
            value, sum_value = __get_p_i_j()
        else:
            value, sum_value = __get_p_i_j_custom()
        value *= sum_value
        value = np.exp(value)
        sum_value = np.exp(sum_value)
        sum_value = dispatch_d_1s(distribution_type, sum_value) * value
        return sum_value
