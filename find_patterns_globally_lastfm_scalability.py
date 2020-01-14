import os
import numpy as np
import pandas as pd
import networkx as nx
import time
import warnings

from collections import defaultdict
from operator import itemgetter
from functools import partial
from multiprocessing import Pool

import scipy.stats as stats
import scipy.sparse as sparse
from scipy import optimize
import scipy.io

from os.path import join

import ICofBlock
from maxent.baseclass.optimization import BGDistr
from utils import memoize, mkdir
import pickle

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

from sklearn.preprocessing import MultiLabelBinarizer
from heapq import nlargest

def load_data(num_attr):
    ud = True
    cwd = os.getcwd()
    file_re = join(cwd, *['datasets','Lastfm', 'user_friends.dat'])
    file_tags = join(cwd, *['datasets','Lastfm', 'tags.dat'])
    file_attr = join(cwd, *['datasets','Lastfm', 'user_taggedartists.dat'])

    # Read the tag dataset
    data_tag = pd.read_csv(file_tags,
                        sep='\t',
                        encoding="latin1")


    tagID = list(data_tag[data_tag.columns[0]])
    tagValue = list(data_tag[data_tag.columns[-1]])

    tag_mapping = dict(zip(tagID,tagValue))

    print(tagID[0])

    # Get all the users
    data_user = pd.read_csv(file_attr,
                    sep='\t',
                    header=1,
                    usecols=[0])

    userID = data_user.values
    print(userID[0])

    # Construct a grpah correspoinding to the friend relationship dataset
    data_re = np.genfromtxt(file_re,
                         names=True,
                         dtype=None,
                         usecols=(0,1))

    print(data_re[0])

    G=nx.Graph()
    G.add_nodes_from(np.unique(userID))
    G.add_edges_from(data_re)
    A = nx.adjacency_matrix(G)

    N = len(G)
    node_mapping=dict(zip(G.nodes(),range(N)))
    # # Construct the user-attributes data frame
    data_attr = np.genfromtxt(file_attr,
                         names=True,
                         dtype=None,
                         usecols=(0, 2))

    # a dicitonary with the keys as the users and values as the tags each user assigned
    attr_dict = {node_mapping[k[0]]: set() for k in data_attr}

    # a dictionary with the keys as the tags and values as the assignment frequency
    # of each tag
    attr_freq_dict = dict.fromkeys(tagValue,0)

    ##
    ## get some most frequent tags
    ##
    for k in data_attr:
        attr_dict[node_mapping[k[0]]].add(tag_mapping[k[-1]])
        attr_freq_dict[tag_mapping[k[-1]]] += 1

    attrs = nlargest(num_attr, attr_freq_dict, key=attr_freq_dict.__getitem__)

    attrs_set = set(attrs)
    for k in range(N):
        attr_dict[k]= attr_dict[k].intersection(attrs_set)

    mlb = MultiLabelBinarizer(classes=attrs)
    attr_rec = mlb.fit_transform(list(attr_dict.values()))
    # print(list(mlb.classes_))

    for k in range(N):
        attr_dict[k] = list(attr_rec[k,:])

    attr_M = pd.DataFrame.from_dict(attr_dict, orient='index',columns=attrs)

    print(attr_M.index)
    print(attr_M.columns)

    return  A, attr_M, ud

def compute_individual_prior(A, ud):
    bg_dist = BGDistr(A, datasource='custom')

    (x_rows, x_columns, rowbeans_index, colbeans_index, _, _) = \
    bg_dist.compute_lambdas_in_a_cooler_way(iterations=1000, verbose=False,
                                            undirected=ud,
                                            is_square_adj=False)
    return x_rows, x_columns, rowbeans_index, colbeans_index

def f_Puv(x, overall_density):
    return np.exp(x+x)/(1+np.exp(x+x))-overall_density

def compute_uniform_prior(A):
    overall_density = A.count_nonzero()/(A.shape[0]*(A.shape[0]-1))
    lamb = optimize.fsolve(f_Puv, 1., args = (overall_density))

    x_rows = np.asarray(lamb)
    x_columns = np.asarray(lamb)
    rowbeans_index = np.array([0]*(A.shape[0]))
    colbeans_index = np.array([0]*(A.shape[0]))
    return x_rows, x_columns, rowbeans_index, colbeans_index

def compute_store(attr_M):
    store = defaultdict(dict)
    for attr in list(attr_M.columns):
        store[attr]['cutPoint'] = [1]
        store[attr]['support'] = []
        support = list(np.where(attr_M[attr]==1)[0])
        store[attr]['support'].append(support)
    return store

def eval_one_attr(attr, store, lambda_dict, current_blocks, block_idx,
                  ic_evaluator, ic_of_blocks):
    result = {}
    result['si'] = -np.inf

    selector_store = store[attr]['support']
    cutPoint_store = store[attr]['cutPoint']
    #
    # print('here')

    for separator_idx, sg_nodes in enumerate(selector_store):
        # for block_idx in range(len(current_blocks)):
        copy_lambda_dict = lambda_dict.copy()
        copy_blocks = current_blocks.copy()
        # get the descrptions from dividing through the cutting point
        # each cutting point will cut the current description into two
        # opposite sides
        to_remove = current_blocks[block_idx]
        sg1 = set(to_remove)&set(sg_nodes)
        sg2 = list(set(to_remove)-sg1)
        sg1 = list(sg1)

        # print('separator_idx',separator_idx)

        if len(sg1) != 0 and len(sg2) != 0:
            copy_blocks[block_idx:block_idx] = [sg1,sg2]
            copy_blocks.remove(to_remove)

            ic_evaluator.end_base = 1.

            IC = 0.
            C = len(copy_blocks)
            affected_idx = [block_idx, block_idx+1]

            # iterate each block
            for r in range(C):
                # print('r', r)
                for c in range(C):
                    # print('c', c)
                    if separator_idx == 0 or (r in affected_idx or c in affected_idx):
                        if r == c:
                            new_lambda, IC_block = \
                            ic_evaluator.get_statistics_SingleGroup(
                                copy_blocks[r], lambda_dict)
                        else:
                            new_lambda, IC_block = \
                            ic_evaluator.get_statistics_BiGroups(
                                copy_blocks[r], copy_blocks[c], lambda_dict)

                        if new_lambda != 0.:
                            new_lambda = float(new_lambda)
                            copy_lambda_dict[new_lambda] = [copy_blocks[r],
                                                            copy_blocks[c]]
                        ic_of_blocks[r,c] = IC_block

            # print('finish one selector')

            IC = np.sum(ic_of_blocks)
            SI = IC/(C*(C+1)/2.+50)
            # print(SI)

            if SI > result['si']:
                result['si'] = SI
                result['lambda_dict'] = copy_lambda_dict
                result['current_blocks'] = copy_blocks
                result['attr'] = attr
                result['separator_idx'] = separator_idx
                result['cut_point'] = cutPoint_store[separator_idx]
                result['block_idx'] = block_idx
    return result

def argmax_SI_over_records(records):
    best_idx = -1
    max_si = -np.inf
    for idx, res in enumerate(records):
        if res['si'] > max_si:
            max_si = res['si']
            best_idx = idx
    return best_idx, max_si

def find_next_selector(store, current_blocks, block_idx, lambda_dict,
                       ic_evaluator, ic_of_blocks):
    with Pool(processes=os.cpu_count()) as pool:
        attr_records = pool.map(partial(eval_one_attr,
                                          store=store,
                                          lambda_dict=lambda_dict,
                                          current_blocks=current_blocks,
                                          block_idx=block_idx,
                                          ic_evaluator=ic_evaluator,
                                          ic_of_blocks=ic_of_blocks),
                                     tuple(store.keys()))
    best_attr, max_si = argmax_SI_over_records(attr_records)
    return attr_records[best_attr], max_si

def summarize(A, store, ud, max_num_selector=1, end_base=5.):
    # compute_prior
    x_rows, x_columns, rowbeans_index, colbeans_index = compute_individual_prior(A,ud)
    # x_rows, x_columns, rowbeans_index, colbeans_index = compute_uniform_prior(A)

    # set the IC evaluator
    G = nx.from_scipy_sparse_matrix(A)
    ic_evaluator = ICofBlock.NwTarget(G, x_rows, x_columns, rowbeans_index,
                                         colbeans_index, end_base)

    # initialize the temp variables
    current_blocks = [list(range(A.shape[0]))]
    lambda_dict = {}
    summary = []
    max_si = -np.inf

    for iter in range(max_num_selector):
        block_records = []
        start = time.time()
        for block_idx in range(len(current_blocks)):
            ic_of_blocks = np.zeros((len(current_blocks)+1, len(current_blocks)+1))
            next_selector, _ = find_next_selector(store, current_blocks, block_idx,
                                           lambda_dict, ic_evaluator,ic_of_blocks)
            block_records.append(next_selector)

        best_cut_id , si = argmax_SI_over_records(block_records)
        # if si > max_si:
        #     max_si = si
        # else:
        #     break
        best_cut = block_records[best_cut_id]
        best_cut['time'] = time.time() - start
        current_blocks = best_cut['current_blocks']
        lamda_dict = best_cut['lambda_dict']

        del best_cut['current_blocks']
        del best_cut['lambda_dict']

        best_cut['iter'] = iter
        print(best_cut)
        summary.append(best_cut)

        # # save the summary in each iteration
        # cwd = os.getcwd()
        # file = join(cwd, *['results', 'Lastfm_no0_itr{:d}_individual.pkl'.format(iter)])
        # f = open(file, 'wb')
        # pickle.dump(summary, f)
        # f.close()

    return summary

def print_summary_info(summary):
    for rec in summary:
        print(('iter: {:d}, attr: {:s}, separator_idx: {:d}, cut_point: {:.4f}, '
               'block_idx: {:d}, time: {:.4f}').format(rec['iter'], rec['attr'],rec['separator_idx'],
													   rec['cut_point'], rec['block_idx'], rec['time']))
def scalability_test():
    testing_list = [5,10,20,40,80,160,320,640,1280,2560,5120,10240]
    search_time = []
    # save key variables
    cwd = os.getcwd()
    file = join(cwd, *['results','scalability_global_varyingNumAttr.pkl'])

    for num_attr in testing_list:
        A, attr_M, ud = load_data(num_attr)
        A = A[::1, ::1]

        # compute selector store
        store = compute_store(attr_M)

        # compute the summary
        summary = memoize(summarize, summary_file,
                          refresh=True)(A, store, ud)

        for rec in summary:
            print(('iter: {:d}, attr: {:s}, separator_idx: {:d}, cut_point: {:.4f}, '
                   'block_idx: {:d}, time: {:.4f}').format(rec['iter'], rec['attr'],rec['separator_idx'],
    													   rec['cut_point'], rec['block_idx'], rec['time']))
            search_time.append(rec['time'])

        Obj = (testing_list,search_time)
        f = open(file,'wb')
        pickle.dump(Obj, f)
        f.close()

    f = open(file,'rb')
    Obj = pickle.load(f)
    f.close()
    print(Obj)

if __name__ == "__main__":
    # supress the scipy warning:
    # RuntimeWarning: The number of calls to function has reached maxfev = 400.
    warnings.simplefilter("ignore")
    summary_file = 'Lastfm_summary.pkl'

    scalability_test()
