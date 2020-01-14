
import numpy as np
import array
import scipy.sparse as sp
import networkx as nx


# class entity(object):
#     """docstring for entity"""

#     def __init__(self):
#         super(entity, self).__init__()
#         self.colind = 0
#         self.rowind = []


class MRD(object):
    """docstring for MRD"""

    def __init__(self, keys={}, data={}, entities_dict={}):
        super(MRD, self).__init__()
        self._keys = keys
        self._data = data
        self._entities_dict = entities_dict

    # Properties

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        self._keys = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def entities_dict(self):
        return self._entities_dict

    @entities_dict.setter
    def entities_dict(self, value):
        self._entities_dict = value

    # Read only propertie

    # Methods

    def create_index(self, filename, header=False, schema=False, separator=','):
        """Create incidence matrix for/from MRD.

        Parameters
        ----------
        filename : string

            specifies the filepath

        header : boolean

            specifies if the file contains a header for entries.

        separator : string

            specifies the separator between entries.

        schema : string

            xml file containing schema of db

        Examples
        --------


        Notes
        -----
        Still need to consider the case in which header=True.
        """

        with open(filename, 'r') as f:
            if header:
                keys = f.readline().strip().split(separator)
                # consider doing a dictionary keys: index in list
                entities_dict = {k: set() for k in keys}
            for l in f:
                [entities_dict[keys[j]].add(element) for j, element in enumerate(l.strip().split(separator))]
            for k in entities_dict.keys():
                # entities_dict[k] = {i: j for i, j in zip(range(len(entities_dict[k])), entities_dict[k])}
                entities_dict[k] = {j: i for i, j in zip(range(len(entities_dict[k])), entities_dict[k])}
            # self.keys = keys
            self._keys = {k: i for i, k in enumerate(keys)}
            self.entities_dict = entities_dict

    def from_file(self, filename, view=None, header=False, separator=','):
        """Create a dictionary of incidence matrixes for/from MRD.

        Returns a dictionary with the primary key chosen by the user and binary incidence matrixes indexed by entity types.

        Based on example role_in_film.mrd, it returns:

        {'pk': 'person', 'role': <2x2 sparse matrix of type '<class 'numpy.int32'>'
        with 3 stored elements in Compressed Sparse Row format>, 'film': <2x2 sparse matrix of type '<class 'numpy.int32'>'
        with 3 stored elements in Compressed Sparse Row format>}

        Parameters
        ----------

        filename: string

            Specifies the filepath

        view: string

            primary key chosen to create relationships matrixes. Byt default it select the first key out of the dictionary.

        header: boolean

            specifying if the file contains a header for entries.

        separator: string

            Specifies the separator between entries.

        See Also
        --------

        create_index

        Examples
        --------

        >>> from baseclass.MRDbase import *
        >>> import os
        >>> cwd = os.getcwd()
        >>> file = os.path.join(cwd, *['examples','movies', 'relationships', 'role_in_film.mrd'])
        >>> db = MRD()
        >>> db.from_file(file, header=True)

        Notes
        -----
        Function still needs lot of improvements
        """

        self.create_index(filename, header, separator)
        if not view:
            view = list(self.keys.keys())[0]  # first random element for the moment
        view_index = self.keys[view]
        to_view = [k for k in self.keys.keys() if k != view]
        offsets = [len(self.entities_dict[k]) for k in self.entities_dict.keys() if k != view]
        Relationship = {'pk': view}  # Specifying the primary key
        n = len(self.entities_dict[view])
        # pdb.set_trace()
        for t, o in zip(to_view, offsets):
            Relationship[t] = incrementalCSRMatrix((n, o), np.int32)  # Dictionary has a key for every available entity

        with open(filename, 'r') as f:
            if header:
                f.readline()  # just remove first line
            for l in f:
                line = l.strip().split(separator)
                row_viewer = line.pop(view_index)
                # row index
                row = self.entities_dict[view][row_viewer]
                for t, ll in zip(to_view, line):
                    col = self.entities_dict[t][ll]
                    # add element only if not already considered.
                    if not ((row, col) in zip(Relationship[t].rows, Relationship[t].cols)):
                        Relationship[t].append(row, self.entities_dict[t][ll], 1)
                    # Note that without the if we collect frequencies of items.
            for t in to_view:
                Relationship[t] = Relationship[t].tocsr()

        self._data = Relationship
        return Relationship


class graphdata(nx.DiGraph):
    """Subclass of DiGraph objects which loads from file and """

    def __init__(self):
        super(graphdata, self).__init__()

    def from_file(self, filename, header=False, separator=',', weighted=False):
        """Store data from graphs txt files.

        Parameters
        ----------
        filename: string specifying the filepath
        header: boolean specifying if the file contains a header for entries.
        separator: specifies the separator between entries.

        See Also
        --------
        related_method_1:

        Examples
        --------

        >>> from baseclass.MRDbase import **
        >>> import os
        >>> cwd = os.getcwd()
        >>> file = os.path.join(cwd, *['examples','graphs', 'DBLP_AA_GCC_edges.txt'])
        >>> g = graphdata()
        >>> g.from_file(file, separator=' ')

        Notes
        -----
        Ulterior things to explain to user.
        """
        edges = []
        # consider also using read_edgelist from networkx module
        # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.readwrite.edgelist.read_edgelist.html
        with open(filename, 'r') as f:
            for l in f:
                edges.append(tuple(l.strip().split(' ')))  # must uniform line split
        if weighted:
            self.add_weighted_edges_from(edges)
        else:
            self.add_edges_from(edges)

    def adj_matrix(self):
        """Create incidence matrix for/from MRD.

        Parameters
        ----------

        See Also
        --------
        related_method_1:

        Examples
        --------

        >>> from baseclass.MRDbase import **
        >>> import os
        >>> cwd = os.getcwd()
        >>> file = os.path.join(cwd, *['examples','graphs', 'DBLP_AA_GCC_edges.txt'])
        >>> g = graphdata()
        >>> g.from_file(file)
        >>> adjacency = g.adj_matrix()


        Notes
        -----
        Ulterior things to explain to user.
        """

        return nx.adjacency_matrix(self)


class incrementalCSRMatrix(object):
    """Class for filling incrementally CSR matrix

    This elegant simple class is a simple hack of this:

    http://maciejkula.github.io/2015/02/22/incremental-sparse-matrices/

    """

    def __init__(self, shape, dtype):
        """Docstring to be defined"""

        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    def append(self, i, j, v):
        m, n = self.shape

        if (i >= m or j >= n):
            raise Exception('Index out of bounds')

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocsr(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return sp.csr_matrix((data, (rows, cols)),
                             shape=self.shape)  # .astype(bool)

    def __len__(self):

        return len(self.data)
