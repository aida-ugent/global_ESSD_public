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
from MPdata_loader import *

import ICofBlock
from maxent.baseclass.optimization import BGDistr
from utils import memoize, mkdir
import pickle

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

from graphviz import Graph
from graphviz import Digraph

def load_data(data_folder, cache_folder):
    raw_data_file = join(data_folder, 'votingDatain13Articles.mat')
    MP_data_cache_file = join(cache_folder, 'MP_data.pkl')
    attr_M_cache_file = join(cache_folder, 'MP_attr.pkl')
    A_cache_file = join(cache_folder, 'MP_A.pkl')

    MP_data = memoize(load_MP_data, MP_data_cache_file,refresh=True)(
                            raw_data_file)
    attr_M = memoize(get_all_votes, attr_M_cache_file, refresh=True)(MP_data)
    A = memoize(get_friends_A, A_cache_file, refresh=True)(MP_data)
    ud = True
    return MP_data, A, attr_M, ud

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
    # “-1， 0， 1”
    # store = defaultdict(dict)
    # for attr in list(attr_M.columns):
    #     store[attr]['cutPoint'] = list(pd.unique(attr_M[attr]))
    #     store[attr]['support'] = []
    #     for cut_point in store[attr]['cutPoint']:
    #         support = list(np.where(attr_M[attr]==cut_point)[0])
    #         store[attr]['support'].append(support)

    # ”>=0, <0"
    store = defaultdict(dict)
    for attr in list(attr_M.columns):
        store[attr]['cutPoint'] = ['>=0','<=0']
        store[attr]['support'] = []

        support = list(np.where(attr_M[attr]>=0)[0])
        store[attr]['support'].append(support)
        support_ = list(np.where(attr_M[attr]<=0)[0])
        store[attr]['support'].append(support_)

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
            SI = IC/(C*(C+1)/2.+C+50)
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

def summarize(A, store, ud, max_num_selector=6, end_base=5.):
    # compute_prior
    # x_rows, x_columns, rowbeans_index, colbeans_index = compute_individual_prior(A,ud)
    x_rows, x_columns, rowbeans_index, colbeans_index = compute_uniform_prior(A)

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

        best_cut = block_records[best_cut_id]

        current_blocks = best_cut['current_blocks']
        lamda_dict = best_cut['lambda_dict']

        del best_cut['current_blocks']
        del best_cut['lambda_dict']

        best_cut['iter'] = iter
        best_cut['time'] = time.time() - start
        print(best_cut)
        summary.append(best_cut)

        # save the summary in each iteration
        cwd = os.getcwd()
        file = join(cwd, *['results', 'MP_no0_itr{:d}_uniform.pkl'.format(iter)])

        f = open(file, 'wb')
        pickle.dump(summary, f)
        f.close()

    return summary

def print_summary_info(summary):
    for rec in summary:
        # print(('iter: {:d}, attr: {:s}, separator_idx: {:d}, cut_point: {:.4f}, '
        #        'block_idx: {:d}, time: {:.4f}').format(rec['iter'], rec['attr'],rec['separator_idx'],
		# 											   rec['cut_point'], rec['block_idx'], rec['time']))
        print(('iter: {:d}, attr: {:s}, separator_idx: {:d}, cut_point: {:s}, '
               'block_idx: {:d}, time: {:.4f}').format(rec['iter'], rec['attr'],rec['separator_idx'],
													   rec['cut_point'], rec['block_idx'], rec['time']))

def get_MPs_info_for_a_sg(MP_data, idx_block, summary, attr_M, store, N):
    parties, MP_party_dict = get_all_parties(MP_data)
    idx_name_dict = get_MP_names(MP_data)
    current_blocks, cuttingPoint_list, block_descrips=get_information_for_blocks(summary, attr_M, store, N)
    party_list = []
    party_names = ['Not applicable','Conservative','Democratic Unionist Party','Green','Independent','Liberal Democrat',
    'Labour','Plaid Cymru','Sinn Fein','Scottish National Party']
    party_mem_dict = defaultdict(list)
    party_num_dict = dict.fromkeys(party_names, 0)

    for MP in current_blocks[idx_block]:
        # print('{:s},   {:s}'.format(idx_name_dict[MP], MP_party_dict[MP]))
        party_mem_dict[MP_party_dict[MP]].append(idx_name_dict[MP])
        party_num_dict[MP_party_dict[MP]] += 1
    return party_mem_dict, party_num_dict

def get_density_matrix(current_blocks, A):
    n = len(current_blocks)
    M = np.zeros((n, n))

    # compute the density for connections between each pair of groups
    for i in range(n):
        for j in range(n):

            sg1_N = len(current_blocks[i])
            sg2_N = len(current_blocks[j])

            # finding the common members in sg1_nodes and sg2_nodes
            com_nodes = list(set(current_blocks[i])&set(current_blocks[j]))
            com_N = len(com_nodes)

            sg_N = sg1_N * sg2_N - com_N

            if sg_N != 0:
                R = np.asarray(current_blocks[i])
                C = np.asarray(current_blocks[j])

                Rows = A[R]
                sg_K = Rows[:,C].count_nonzero()

                M[i][j] = sg_K/sg_N
    return M

def get_information_for_blocks(summary, attr_M, store, N):
    current_blocks = [list(range(N))]
    block_descrips = [[]]
    cuttingPoint_list = []
    sortedCutPoint_list = [0]

    # get the description for each block
    for rec in summary:
        sg_nodes = store[rec['attr']]['support'][rec['separator_idx']]

        toAdd_descrip1 =  'attr: {:s}, value: {:s}'.format(rec['attr'],
                                                rec['cut_point'])
        toAdd_descrip2 = 'NOT attr: {:s}, value: {:s}'.format(rec['attr'],
                                                rec['cut_point'])

        # cut_type = store[rec['attr']]['cutPoint'][rec['separator_idx']]
        # toAdd_descrip1 = '{:s} {:s}'.format(rec['attr'], cut_type)
        # toAdd_descrip2 = 'not {:s} {:s}'.format(rec['attr'],cut_type)

        # get the descrptions from dividing through the cutting point
        # each cutting point will cut the current description into two opposite sides
        to_remove = current_blocks[rec['block_idx']]
        sg1 = set(to_remove)&set(sg_nodes)
        sg2 = list(set(to_remove)-sg1)
        sg1 = list(sg1)
        # print(sg1[:10])

        # sort members in new blocks according to their values for this cutting attribute
        # attr_sg1 = attr_proj[np.asarray(sg1)][:, rec['attr']]
        # attr_sg2 = attr_proj[np.asarray(sg2)][:,rec['attr']]

        attr_sg1 = np.array(attr_M[rec['attr']].iloc[np.asarray(sg1)])
        attr_sg2 = np.array(attr_M[rec['attr']].iloc[np.asarray(sg2)])

        # print(attr_sg1.todense()[:10])
        sorted_attr_sg1 = sorted(zip(sg1, attr_sg1), key=itemgetter(1), reverse=False)
        sg1 = [e[0] for e in sorted_attr_sg1]
        # print(sg1[:10])

        sorted_attr_sg2 = sorted(zip(sg2, attr_sg2), key=itemgetter(1), reverse=False)
        sg2 = [e[0] for e in sorted_attr_sg2]

        # update current_blocks        # attr_sg1 = attr_proj[np.asarray(sg1)][:, rec['attr']]
        # attr_sg2 = attr_proj[np.asarray(sg2)][:,rec['attr']]
        current_blocks[rec['block_idx']:rec['block_idx']] = [sg1,sg2]
        current_blocks.remove(to_remove)

        # update cuttingPoint_list
        cuttingPoint = sortedCutPoint_list[rec['block_idx']] + len(sg1)
        sortedCutPoint_list[rec['block_idx']+1:rec['block_idx']+1] = [cuttingPoint]
        cuttingPoint_list.append(cuttingPoint)

        # update block_descrips
        to_cut = block_descrips[rec['block_idx']]
        sg1_descrip = to_cut.copy()
        sg2_descrip = to_cut.copy()
        sg1_descrip.append(toAdd_descrip1)
        sg2_descrip.append(toAdd_descrip2)
        block_descrips[rec['block_idx']:rec['block_idx']] = [sg1_descrip, sg2_descrip]
        block_descrips.remove(to_cut)

    return current_blocks, cuttingPoint_list, block_descrips,

def visualize_by_collapsedNodes(summary, attr_M, store, A, N):
    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary,attr_M, store, N)

    # # filter out groups with less than 10 members
    # updated_blocks = []
    # updated_descrips = []
    #
    # for i in range(len(current_blocks)):
    #     if len(current_blocks[i]) > 10:
    #         updated_blocks.append(current_blocks[i])
    #         updated_descrips.append(block_descrips[i])
    #
    # current_blocks = updated_blocks
    # block_descrips = updated_descrips

    # descrips_list = ['I1: against or abstention'+'\n'+'I10 V3: against or abstention'
    # +'\n'+'I10 V4: against or abstention', 'I1: against or abstention'+'\n'+'I10 V3: against or abstention'
    # +'\n'+'I10 V4: in favour','I1: against or abstention'+'\n'+'I10 V3: in favour',
    # 'I1: in favour'+'\n'+'I7 V4: in favour or abstention', 'I1: in favour'+'\n'+'I7 V4: against']

    descrips_list = ['I1 = -1 or 0'+'\n'+'I10 V3 = -1 or 0'
    +'\n'+'I10 V4 = -1 or 0', 'I1 = -1 or 0'+'\n'+'I10 V3 = -1 or 0'
    +'\n'+'I10 V4 = 1','I1 = -1 or 0'+'\n'+'I10 V3 = 1',
    'I1 = 1'+'\n'+'I7 V4 = 1 or 0', 'I1 = 1'+'\n'+'I7 V4 = -1']


    # compute the connectivity density between each pair of blocks
    density_M = get_density_matrix(current_blocks, A)

    # a circular layout visualization
    # g = Digraph('G','collapsedNetwork_dblp.gv')
    g = Graph('G','collapsedNetwork_MP.gv')
    n = len(current_blocks)
    widths = [0.07*len(current_blocks[i])**(1/2)+.1 for i in range(n)]

    g.attr(layout = 'circo',splines ='spline', nodesep='2.', mindist='4.5')

    for i in range(n):
        # g.attr('node', shape='circle', fixedsize='true', fontsize='20',fontcolor='{h:} 0.8 0.8'.format(h=i/n),style='filled',\
        #       fillcolor='{h:} 0.6 0.8'.format(h=i/n),width=str(widths[i]),\
        #        xlabel=descrip_list[i]+str(len(current_blocks[i])))
        g.attr('node', shape='circle', fixedsize='true', fontsize='30',style='filled',\
              fillcolor='#9ecae1',width=str(widths[i]),\
               xlabel=descrips_list[i])
#                 pos='1,str(sum(widths[:i+1])+1.*i)')
#         g.node('cluster %d'% i)
        g.node(str(i+1), label='')

    for i in range(n):
        for j in range(n):
            g.attr('edge', penwidth = str(20*density_M[i][j]), color='#7d7d7d50', arrowsize = str(1), arrowhead='normal')
#             g.edge('cluster %d'% i, 'cluster %d'% j)
            g.edge(str(i+1), str(j+1))

    g.view()

def visualize_by_heatmap(MP_data,summary, attr_M, store, A, N):
    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary,attr_M, store, N)

    descrips_list = ['I1 = -1 or 0'+'\n'+'I10 V3 = -1 or 0'
    +'\n'+'I10 V4 = -1 or 0', 'I1 = -1 or 0'+'\n'+'I10 V3 = -1 or 0'
    +'\n'+'I10 V4 = 1','I1 = -1 or 0'+'\n'+'I10 V3 = 1',
    'I1 = 1'+'\n'+'I7 V4 = 1 or 0', 'I1 = 1'+'\n'+'I7 V4 = -1']

    # compute the connectivity density between each pair of blocks
    density_M = get_density_matrix(current_blocks, A)
    print(density_M)

    fig, ax = plt.subplots()
    im = ax.imshow(density_M, cmap = "YlGn")

    ax.set_xticks(np.arange(len(current_blocks)))
    ax.set_yticks(np.arange(len(current_blocks)))

    ax.set_xticklabels(list(range(1,len(current_blocks)+1)))
    ax.set_yticklabels(descrips_list)

    # plt.setp(ax.get_xticklabels(), rotation=15, ha="right",rotation_mode="anchor")
    # plt.setp(ax.get_xticklabels(), rotation=15, ha="right",rotation_mode="anchor")
    for i in range(len(current_blocks)):
        for j in range(len(current_blocks)):
            text = ax.text(j,i,round(density_M[i,j],4),ha="center", va="center")

    plt.show()

def visualize_by_densityMatrix(summary, store, A, attr_M, N):
    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary, attr_M, store, N)

    # further process 'block_descrips' to make it look more legible
    stringList_descrips = []
    for list_descrip in block_descrips:
        descrip = ''
        for str_descrip in list_descrip:
            descrip = descrip + str_descrip + '\n'
        stringList_descrips.append(descrip)

    sorted_ids = np.hstack(current_blocks)

    A = A[sorted_ids][:,sorted_ids]

    fig = plt.figure(figsize=(10, 10)) # in inches
    plt.imshow(A.todense(),
            cmap="Greys",
            interpolation="none")

    cm_subsection = np.linspace(0,1,len(summary))
    colors = [ cm.jet(x) for x in cm_subsection ]
    shift = 0
    ax = plt.gca()

    for k in range(len(summary)):
        shift = cuttingPoint_list[k]
#         print(shift)
#         print(len(current_blocks[k]))

        x, y = np.array([[shift, shift],[0, N]])
        s, t = np.array([[0,N],[shift, shift]])

        V_line = mlines.Line2D(x , y, lw=1.5, alpha=0.6, color = colors[k], label = 'cut '+ str(k)+ '\n ' + stringList_descrips[k])
        H_line = mlines.Line2D(s, t, lw=1.5, alpha=0.6, color = colors[k])
        ax.add_line(V_line)
        ax.add_line(H_line)

    ax.legend(loc='upper right')
    # plt.savefig(filename)
    plt.show()

def visualize_by_onehistogram(MP_data, attr_M, summary, store, N):
    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary, attr_M, store, N)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['black', '#0087dc', '#D46A4C', '#528D6B', '#DDDDDD','#FAA61A','#DC241f','#008142','#326760','#FDF38E']
    width = 0.25
    ind = 2

    custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]
    labels=['Not applicable','Conservative','Democratic Unionist Party','Green','Independent','Liberal Democrat',
    'Labour','Plaid Cymru','Sinn Fein','Scottish National Party']

    # descrips_list = ['I1: against or abstention'+'\n'+'I10 V3: against or abstention'
    # +'\n'+'I10 V4: against or abstention', 'I1: against or abstention'+'\n'+'I10 V3: against or abstention'
    # +'\n'+'I10 V4: in favour','I1: against or abstention'+'\n'+'I10 V3: in favour',
    # 'I1: in favour'+'\n'+'I7 V4: in favour or abstention', 'I1: in favour'+'\n'+'I7 V4: against']

    descrips_list = ['I1 = -1 or 0'+'\n'+'I10 V3 = -1 or 0'
    +'\n'+'I10 V4 = -1 or 0', 'I1 = -1 or 0'+'\n'+'I10 V3 = -1 or 0'
    +'\n'+'I10 V4 = 1','I1 = -1 or 0'+'\n'+'I10 V3 = 1',
    'I1 = 1'+'\n'+'I7 V4 = 1 or 0', 'I1 = 1'+'\n'+'I7 V4 = -1']

    # self-defined block order
    orders = [4,3,2,1,0]

    for i in orders:
        xloc = 0
        j=0

        party_mem_dict, party_num_dict = get_MPs_info_for_a_sg(MP_data, i, summary, attr_M, store, N)

        for par in list(party_num_dict.keys()):
            num_members = party_num_dict[par]
            rect = ax.barh(ind, num_members, width, left=xloc, color=colors[j])
            xloc += num_members
            j+=1
        plt.text(xloc,ind,str(len(current_blocks[i])))
        plt.text(10, ind + width + 0.03, descrips_list[i], fontsize=11)
        ind += 1

    ax.margins(x=0.5)
    ax.legend(custom_lines, labels,loc='best')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # supress the scipy warning:
    # RuntimeWarning: The number of calls to function has reached maxfev = 400.
    warnings.simplefilter("ignore")
    data_folder = './datasets/'
    cache_folder = mkdir('./cache/')
    summary_file = 'MP_summary.pkl'
    max_num_selector = 6

    MP_data, A, attr_M, ud = load_data(data_folder, cache_folder)
    A = A[::1, ::1]

    # compute selector store
    store = compute_store(attr_M)
    print(store)
    #
    # compute the summary
    summary = memoize(summarize, summary_file,
                      refresh=True)(A, store, ud)

    print_summary_info(summary)
    for i in range(5):
        party_mem_dict, party_num_dict = get_MPs_info_for_a_sg(MP_data, i, summary, attr_M, store, A.shape[0])
        print(party_mem_dict, party_num_dict)

    # # visualize the summary results
    # # visualize_by_densityMatrix(summary, store, A, attr_M, A.shape[0])
    # visualize_by_collapsedNodes(summary, attr_M, store, A, A.shape[0])
    # visualize_by_heatmap(MP_data, summary, attr_M, store, A, A.shape[0])
    # visualize_by_onehistogram(MP_data,attr_M, summary, store, A.shape[0])
