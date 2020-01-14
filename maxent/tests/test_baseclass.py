# import pytest
import numpy.testing as npt
import os
import numpy as np
from maxent.baseclass.MRDbase import *
# import pdb

# run pytest test_baseclass.py -s to see message printed on screen


class TestBase(object):
    """docstring for TestBase"""

    def test_keys_loading(self):
        # dictionaries are not ordered, unit tests must be smarter.
        # role_testmatrix = np.array(([1, 0], [1, 1], [0, 1], [1, 1]))
        # film_testmatrix = np.array(([1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]))
        cwd = os.getcwd()
        parent_dir = os.path.dirname(cwd)
        file = os.path.join(*[parent_dir, 'examples', 'movies', 'role_in_film.mrd'])
        db = MRD()
        db.from_file(file, header=True)
        header = ['person', 'role', 'film']
        keys_dict = {}
        keys_dict['person'] = ['Quentin Tarantino', 'Ethan Coen', 'Joel Coen', 'Dario Argento']
        keys_dict['role'] = ['director', 'writer']
        keys_dict['film'] = ['Fargo', 'True Grit', 'Suspiria', 'From Dusk Till Dawn']
        # the keys in entities_dict seems to be ordered in different way according to when they are
        # npt.assert_array_equal(np.asarray(db.data['role'].todense()), role_testmatrix)
        # npt.assert_array_equal(np.asarray(db.data['film'].todense()), film_testmatrix)
        assert db.data['pk'] == 'person'
        for k in db.entities_dict.keys():
            assert k in header
            print('Check for: ' + k)
            for kk in db.entities_dict[k].keys():
                assert kk in keys_dict[k]
                print(kk + ' seems to be here.')

    def test_rows(self):
        def out_row(k, kk):
            return db.data[k].todense()[db.entities_dict['person'][kk], :]
        cwd = os.getcwd()
        parent_dir = os.path.dirname(cwd)
        file = os.path.join(*[parent_dir, 'examples', 'movies', 'role_in_film.mrd'])
        db = MRD()
        db.from_file(file, header=True)
        # still not sure about order.
        # pdb.set_trace()
        assert np.all(out_row('role', 'Ethan Coen') == np.array(([1, 1])))
        assert np.all(out_row('role', 'Quentin Tarantino') == np.array(([1, 0]))) # might fail because the order of column indexes may be different
        assert np.all(out_row('film', 'Dario Argento') == np.array(([0, 0, 1, 0])))
