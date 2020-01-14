import operator
import os
import time
import warnings
from collections import defaultdict
from operator import itemgetter
from functools import partial
from multiprocessing import Pool
from os.path import join

import networkx as nx
import numpy as np

import scipy.stats as stats
import scipy.sparse as sparse
from scipy import optimize

import ICofBlock
from load_dblp_data import (load_dblp_data, load_dblp_data_affs, compute_adj, compute_attributes,
                            compute_LSA)
from maxent.baseclass.optimization import BGDistr
from utils import memoize, mkdir
import pickle

import pycountry

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from graphviz import Graph
from graphviz import Digraph


def load_data(data_name, data_folder, cache_folder, subsample_step=8):
    raw_data_file = join(data_folder, 'dblp_papers_v11.txt')
    paper_records_cache_file = join(cache_folder, 'dblp_paper_records_affs.pkl')
    dblp_data_cache_file = join(cache_folder, 'dblp_data_affs.pkl')
    adj_A_cache_file = join(cache_folder, 'dblp_adj_A_affs.pkl')

    dblp_data, att_A = memoize(load_dblp_data_affs, dblp_data_cache_file,refresh=True)(data_name,
                            raw_data_file, paper_records_cache_file,
                            subsample_step)
    adj_A = memoize(compute_adj, adj_A_cache_file, refresh=True)(dblp_data)
    return dblp_data, adj_A, att_A

def load_full_data(data_name, data_folder, cache_folder, subsample_step=8):
    raw_data_file = join(data_folder, 'dblp_papers_v11.txt')
    paper_records_cache_file = join(cache_folder, 'dblp_paper_records_affs.pkl')
    dblp_data_cache_file = join(cache_folder, 'dblp_data_affs.pkl')
    adj_A_cache_file = join(cache_folder, 'dblp_adj_A_affs.pkl')

    dblp_data, att_A = memoize(load_dblp_data_affs, dblp_data_cache_file,refresh=True)(data_name,
                            raw_data_file, paper_records_cache_file,
                            subsample_step)
    adj_A = memoize(compute_adj, adj_A_cache_file, refresh=True)(dblp_data)
    return dblp_data, adj_A, att_A

def compute_prior(adj_A):
    bg_dist = BGDistr(adj_A, datasource='custom')

    (x_rows, x_columns, rowbeans_index, colbeans_index, _, _) = \
    bg_dist.compute_lambdas_in_a_cooler_way(iterations=1000, verbose=False,
                                            undirected=False,
                                            is_square_adj=False)
    return x_rows, x_columns, rowbeans_index, colbeans_index

def f_Puv(x, overall_density):
    return np.exp(x+x)/(1+np.exp(x+x))-overall_density

def compute_uniform_prior(adj_A):
    overall_density = adj_A.count_nonzero()/(adj_A.shape[0]*(adj_A.shape[0]-1))
    lamb = optimize.fsolve(f_Puv, 1., args = (overall_density))

    x_rows = np.asarray(lamb)
    x_columns = np.asarray(lamb)
    rowbeans_index = np.array([0]*(adj_A.shape[0]))
    colbeans_index = np.array([0]*(adj_A.shape[0]))

    return x_rows, x_columns, rowbeans_index, colbeans_index

def compute_store(att_A):
    store = defaultdict(dict)
    for attr in list(att_A.columns):
        store[attr]['cutPoint'] = [1]
        store[attr]['support'] = []
        support = list(np.where(att_A[attr]==1)[0])
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
            SI = IC/(C*(C+1)/2.+C+50.)
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

def summarize(adj_A, store, max_num_selector=6, end_base=5.):
    # compute_prior
    # x_rows, x_columns, rowbeans_index, colbeans_index = compute_prior(adj_A)
    x_rows, x_columns, rowbeans_index, colbeans_index = compute_uniform_prior(adj_A)

    # set the IC evaluator
    G = nx.from_scipy_sparse_matrix(adj_A)
    ic_evaluator = ICofBlock.NwTarget(G, x_rows, x_columns, rowbeans_index,
                                         colbeans_index, end_base)

    # initialize the temp variables
    current_blocks = [list(range(adj_A.shape[0]))]
    lambda_dict = {}
    summary = []
    max_si = -np.inf
    for iter in range(max_num_selector):
        block_records = []
        start = time.time()
        for block_idx in range(len(current_blocks)):
            ic_of_blocks = np.zeros((len(current_blocks)+1, len(current_blocks)+1))
            next_selector,_ = find_next_selector(store, current_blocks, block_idx,
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
        file = join(cwd, *['results', 'full_dblp_affs_itr{:d}_uniform.pkl'.format(iter)])

        f = open(file, 'wb')
        pickle.dump(summary, f)
        f.close()

    return summary

def print_summary_info(summary):
    for rec in summary:
        print(('iter: {:d}, attr: {:s}, separator_idx: {:d}, cut_point: {:.4f}, '
               'block_idx: {:d}, time: {:.4f}').format(rec['iter'], rec['attr'],rec['separator_idx'],
													   rec['cut_point'], rec['block_idx'], rec['time']))

def print_affiliation_info(idx_block, dblp_data, summary, attr_proj, store, adj_A, N):
    vid_venue_dict = {'1203999783':'IJCAI',  '1184914352': 'AAAI', '1180662882': 'ICML', '1127325140': 'NIPS',
     '2584161585': 'ICLR', '1163988186': 'ICDE','1133523790': 'VLDB', '1175089206': 'SIGMOD', '1165285842': 'ICDT',
      '1184151122': 'PODS', '1140684652': 'SIGIR', '1135342153': 'WWW', '1194094125': 'CIKM', '1180513217': 'ECIR',
      '1130985203': 'KDD', '1141769385': 'ECML-PKDD', '1120384002': 'WSDM', '1178327129': 'PAKDD','1183478919': 'ICDM',
      '1142743330': 'SDM'}

    paper_records = dblp_data['paper_records']
    pid_id_dict = dblp_data['pid_id_dict']
    id_pid_dict = {pid_id_dict[pid]: pid for pid in list(pid_id_dict.keys())}

    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary,attr_proj, store, N)
    nodes_block = current_blocks[idx_block]
    counter = 0

    print(block_descrips[idx_block])
    for id in nodes_block:
        print('-----------------------------------------------------------')
        pid = id_pid_dict[id]
        venue_id = paper_records[pid]['venue']['id']
        print(paper_records[pid]['authors'])
        print(vid_venue_dict[venue_id])
        print(paper_records[pid]['title'])
        print('')
        counter += 1
        if counter > 100:
            break

def get_density_matrix(current_blocks, Adj_A):
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

                Rows = Adj_A[R]
                sg_K = Rows[:,C].count_nonzero()

                M[i][j] = sg_K/sg_N
    return M

def get_information_for_blocks(summary, attr_proj, store, N):
    current_blocks = [list(range(N))]
    block_descrips = [[]]
    cuttingPoint_list = []
    sortedCutPoint_list = [0]

    # get the description for each block
    for rec in summary:
        sg_nodes = store[rec['attr']]['support'][rec['separator_idx']]
        # toAdd_descrip1 =  'attr: {:s}, separator_idx: {:d}'.format(rec['attr'],
        #                                         rec['separator_idx'])
        # toAdd_descrip2 = 'NOT attr: {:s}, separator_idx: {:d}'.format(rec['attr'],
        #                                         rec['separator_idx'])

        toAdd_descrip1 = '{:s} = 1'.format(rec['attr'])
        toAdd_descrip2 = '{:s} = 0'.format(rec['attr'])

        # get the descrptionsprint(paper_records[id_pid_dict[0]]['authors']) from dividing through the cutting point
        # each cutting point will cut the current description into two opposite sides
        to_remove = current_blocks[rec['block_idx']]
        sg1 = set(to_remove)&set(sg_nodes)
        sg2 = list(set(to_remove)-sg1)
        sg1 = list(sg1)
        print(sg1[:10])

        # sort members in new blocks according to their values for this cutting attribute
        # attr_sg1 = attr_proj[np.asarray(sg1)][:, rec['attr']]
        # attr_sg2 = attr_proj[np.asarray(sg2)][:,rec['attr']]

        attr_sg1 = np.array(attr_proj[rec['attr']].iloc[np.asarray(sg1)])
        attr_sg2 = np.array(attr_proj[rec['attr']].iloc[np.asarray(sg2)])

        # print(attr_sg1.todense()[:10])
        sorted_attr_sg1 = sorted(zip(sg1, attr_sg1), key=operator.itemgetter(1), reverse=False)
        sg1 = [e[0] for e in sorted_attr_sg1]
        print(sg1[:10])

        sorted_attr_sg2 = sorted(zip(sg2, attr_sg2), key=operator.itemgetter(1), reverse=False)
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

def visualize_by_collapsedNodes(summary, attr_proj, store, adj_A, N):
    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary,attr_proj, store, N)

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

    # further process 'block_descrips' to make it look more legible
    stringList_descrips = []
    for list_descrip in block_descrips:
        descrip = ''
        for str_descrip in list_descrip:
            descrip = descrip + str_descrip + '\n'
        stringList_descrips.append(descrip)


    # compute the connectivity density between each pair of blocks
    density_M = get_density_matrix(current_blocks, adj_A)

    # a circular layout visualization
    g = Digraph('G','collapsedNetwork_dblp.gv')
    n = len(current_blocks)

    widths = [0.07*len(current_blocks[i])**(1/2)+.1 for i in range(n)]

    g.attr(layout = 'circo',splines ='spline', nodesep='2.', mindist='4.5')

    for i in range(n):
        g.attr('node', shape='circle', fixedsize='true', fontsize='20',fontcolor='{h:} 0.8 0.8'.format(h=i/n),style='filled',\
              fillcolor='{h:} 0.6 0.8'.format(h=i/n),width=str(widths[i]),\
               xlabel=stringList_descrips[i]+str(len(current_blocks[i])))
#                 pos='1,str(sum(widths[:i+1])+1.*i)')
#         g.node('cluster %d'% i)
        g.node(str(i+1), label='')

    for i in range(n):
        for j in range(n):
            g.attr('edge', penwidth = str(10000*density_M[i][j]), color='#7d7d7d50', arrowsize = str(1), arrowhead='normal')
#             g.edge('cluster %d'% i, 'cluster %d'% j)
            g.edge(str(i+1), str(j+1))

    g.view()

def visualize_by_heatmap(dblp_data,summary, attr_proj, store, adj_A, N):
    paper_records = dblp_data['paper_records']
    pid_id_dict = dblp_data['pid_id_dict']
    id_pid_dict = {id: pid for pid, id in pid_id_dict.items()}

    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary,attr_proj, store, N)

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

    # further process 'block_descrips' to make it look more legible
    stringList_descrips = []
    for list_descrip in block_descrips:
        descrip = ''
        for str_descrip in list_descrip:
            descrip = descrip + str_descrip + '\n'
        stringList_descrips.append(descrip)

    # compute the connectivity density between each pair of blocks
    density_M = get_density_matrix(current_blocks, adj_A)
    print(density_M)

    fig, ax = plt.subplots()
    im = ax.imshow(density_M, cmap = "YlGn")

    ax.set_xticks(np.arange(len(current_blocks)))
    ax.set_yticks(np.arange(len(current_blocks)))

    ax.set_xticklabels(stringList_descrips)
    ax.set_yticklabels(stringList_descrips)

    # plt.setp(ax.get_xticklabels(), rotation=15, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right",rotation_mode="anchor")
    for i in range(len(current_blocks)):
        for j in range(len(current_blocks)):
            text = ax.text(j,i,round(density_M[i,j],5),ha="center", va="center")

    plt.show()

def visualize_by_densityMatrix(summary, store, adj_A, attr_proj, N):
    current_blocks, cuttingPoint_list, block_descrips = get_information_for_blocks(summary, attr_proj, store, N)

    # further process 'block_descrips' to make it look more legible
    stringList_descrips = []
    for list_descrip in block_descrips:
        descrip = ''
        for str_descrip in list_descrip:
            descrip = descrip + str_descrip + '\n'
        stringList_descrips.append(descrip)

    sorted_ids = np.hstack(current_blocks)

    adj_A = adj_A[sorted_ids][:,sorted_ids]

    fig = plt.figure(figsize=(10, 10)) # in inches
    plt.imshow(adj_A.todense(),
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


if __name__ == "__main__":
    # supress the scipy warning:
    # RuntimeWarning: The number of calls to function has reached maxfev = 400.
    warnings.simplefilter("ignore")

    data_name = 'dblp_four_areas'
    data_folder = './datasets/'
    cache_folder = mkdir('./cache/')
    summary_file = 'dblp_summary.pkl'
    max_num_selector = 6

    # dblp_data, adj_A, att_A = load_data(data_name, data_folder, cache_folder)
    dblp_data, adj_A, att_A = load_full_data(data_name, data_folder, cache_folder)
    adj_A = adj_A[::1, ::1]

    print(adj_A.shape[0], np.sum(adj_A.todense()))

    paper_records = dblp_data['paper_records']
    pid_id_dict = dblp_data['pid_id_dict']
    id_pid_dict = {pid_id_dict[pid]: pid for pid in list(pid_id_dict.keys())}
    print(paper_records[id_pid_dict[0]]['authors'])
    a = (att_A.iloc[0]==1)
    for name, row in a.items():
        print(name, row)

    # assert False

    # compute selector store
    store = compute_store(att_A)
    #
    # compute the summary
    summary = memoize(summarize, summary_file,
                      refresh=True)(adj_A, store)

    print_summary_info(summary)

    # # # visualize the summary results
    # # visualize_by_densityMatrix(summary, store, adj_A, att_A, adj_A.shape[0])
    # visualize_by_collapsedNodes(summary, att_A, store, adj_A, adj_A.shape[0])
    # visualize_by_heatmap(dblp_data, summary, att_A, store, adj_A, adj_A.shape[0])
