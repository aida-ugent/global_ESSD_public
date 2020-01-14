import numpy as np
import pandas as pd
import networkx as nx
import pysubgroupx as psx
import scipy

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy import optimize

from sklearn.preprocessing import MultiLabelBinarizer

# from maxent.baseclass.optimization_block import *
# from maxent.baseclass.MRDbase import *
from maxent.baseclass import optimization_block
from heapq import nlargest

import pickle

import time
import os
import operator

from visualisation import *
from get_graph_and_attributes import *



# -----------------------------------------------------------------------------
# Get a graph and the user-attributes data frame from a particular dataset
# -----------------------------------------------------------------------------

dataset = input('Which dataset? \n \
Caltech36(Type ct), Reed98(Type r)')

if dataset == "ct":
    G, data, name = from_facebook100('Caltech36.mat')
    ud=True
elif dataset == "r":
    G, data, name = from_facebook100('Reed98.mat')
    ud=True
else:
    print("Invalid option.")

# adjacency matrix
total_A = nx.adjacency_matrix(G)

# -----------------------------------------------------------------------------
# Running choice
# -----------------------------------------------------------------------------

choice = input('Continue from a stored session (Type 1) \n \
or start a new search ? (Type 2)')

if choice == "1":
    cwd = os.getcwd()
    file = os.path.join(cwd, *['results','bi_' + name +'_results.pkl'])

    f = open(file,'rb')
    Obj = pickle.load(f)
    f.close()

    (lambda_sum,lambda_dict,bg_time, search_time, Result) = Obj
    result = Result[-1]
    # print(result)


    target = psx.GTarget_with_BlockCons(G, lambda_sum, lambda_dict, ud)
    searchspace = psx.createSelectors(data)
    # print(searchspace)
    # print(len(searchspace))
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[6,8],depth=2,qf=psx.SubjectiveInterestingness())

elif choice == "2":
    #
    # Initial background distribution
    #
    n = len(G)
    to_incor_attrs = ['year', 'dorm/house']
    label_list = []

    for i in to_incor_attrs:
        label_list.append(data[i])

    bd_graph = optimization_block.BGDistr(total_A, datasource='custom')
    bg_time_s = time.time()
    lambda_sum = bd_graph.compute_lambdas_in_a_cooler_way(label_lists=label_list,iterations=1000,verbose=False,undirected=ud)
    bg_time =  time.time() - bg_time_s

    # -------------------------------------------------------------------------
    # Check row and column lambdas

    P_M = np.exp(lambda_sum)/(1. + np.exp(lambda_sum))

    # The probability of self-edge is 0
    np.fill_diagonal(P_M, 0.0)

    adjM = total_A.todense()
    diff_row = np.sum(P_M,axis=0).squeeze()-np.sum(adjM, axis=0).squeeze()
    diff_col = np.sum(P_M,axis=1).squeeze()-np.sum(adjM, axis=1).squeeze()

    print(np.shape(np.sum(P_M,axis=1)))
    print(np.shape(np.sum(adjM,axis=1).squeeze()))
    print(diff_row[:,:20])
    print(diff_col[:,:20])

    #
    # Check block lambdas
    #
    to_check_attr = ['dorm/house']
    to_check_selectors = []
    lambda_dict = {}

    target = psx.GTarget_with_BlockCons(G, lambda_sum, lambda_dict, ud)

    for attr_name in to_check_attr:
        for val in pd.unique(data[attr_name]):
            sel = psx.subgroup.NominalSelector(attr_name, val)
            to_check_selectors.append(sel)

    # print(to_check_selectors)

    sg1 =  psx.subgroup.Subgroup(target, to_check_selectors[0])
    sg2 =  psx.subgroup.Subgroup(target, to_check_selectors[1])

    sg1_instances = sg1.subgroupDescription.covers(data)
    sg1_nodes = np.where(sg1_instances)[0]

    sg2_instances = sg2.subgroupDescription.covers(data)
    sg2_nodes = np.where(sg2_instances)[0]

    block_sum_act = total_A[sg1_nodes][:,sg2_nodes].count_nonzero()
    print(block_sum_act)

    block_sum_exp = np.sum(P_M[sg1_nodes][:,sg2_nodes])
    print(block_sum_exp)


    #
    # The first search
    #

    lambda_dict = {}
    Result = []

    target = psx.GTarget_with_BlockCons(G, lambda_sum, lambda_dict, ud)

    searchspace = psx.createSelectors(data)
    # searchspace = psx.createSelectors(data,ignore=['high school','major','second major'])
    print(searchspace)
    print(len(searchspace))

    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[6,8],depth=2,qf=psx.SubjectiveInterestingness())

    search_time = []
    search_time_s = time.time()
    result = psx.BeamSearch(beamWidth=10).execute_in_bicases_constraints(task)
    search_time.append(time.time() - search_time_s)

    Result.append(result)
    # print(result)

    # save key variables
    cwd = os.getcwd()
    file = os.path.join(cwd, *['results','bi_' + name +'_results.pkl'])

    Obj = (lambda_sum,lambda_dict,bg_time, search_time,Result)
    f = open(file,'wb')
    pickle.dump(Obj, f)
    f.close()

else:
    print("Invalid option.")

#
# #
def Connecting_Prob_BiCases(sg1_nodes, sg2_nodes, lambda_dict):
    '''Computing the current connecting probability of edges between two subgroups'''
    sg1_N = len(sg1_nodes)
    sg2_N = len(sg2_nodes)

    # finding the common members in sg1_nodes and sg2_nodes
    com_nodes = list(set(sg1_nodes)&set(sg2_nodes))

    com_N = len(com_nodes)

    R = np.asarray(sg1_nodes)
    C = np.asarray(sg2_nodes)

    # The lower bound of the number of edges between sg1 and sg2
    sg_K = target.Adj_M[R][:,C].count_nonzero()   # rows point to cols

    sg_lambda_sum = target.lambda_sum[sg1_nodes][:,sg2_nodes]
    P_M = np.exp(sg_lambda_sum)/(1. + np.exp(sg_lambda_sum))

    # The probability of self-edge is 0
    for i in com_nodes:
        P_M[sg1_nodes.index(i), sg2_nodes.index(i)] = 0.

    for i in lambda_dict.keys():

        com_members1 = list(set(target.lambda_dict[i][0])&set(sg1_nodes))
        com_members2 = list(set(target.lambda_dict[i][1])&set(sg2_nodes))

        if com_members1 != [] and com_members2 != [] :

            com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members1])
            com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members2])

            lambda_M = np.ones((sg1_N, sg2_N))
            lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(i)

            odds_M = P_M * lambda_M
        else:
            odds_M = P_M

        if target.undirected:

            com_members1 = list(set(target.lambda_dict[i][1])&set(sg1_nodes))
            com_members2 = list(set(target.lambda_dict[i][0])&set(sg2_nodes))

            if com_members1 != [] and com_members2 != [] :

                com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members1])
                com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members2])
                lambda_M = np.ones((sg1_N, sg2_N))
                lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(i)

                odds_M = odds_M * lambda_M


        P_M = odds_M / (1. - P_M + odds_M)

    return (P_M, sg_K)



def f_constraint(x, P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K):
    '''Computing the lambda'''
    lambda_M = np.exp(x)*np.ones((sg1_N,sg2_N))
    update_odds_M = P_M * lambda_M

    if target.undirected:
        com_members = list(set(sg1_nodes)&set(sg2_nodes))

        if com_members != [] :

            com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members])
            com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members])

            lambda_M = np.ones((sg1_N, sg2_N))
            lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(x)

            update_odds_M = update_odds_M * lambda_M

    P_M = update_odds_M / (1. - P_M + update_odds_M)

    sg_P = np.sum(P_M)

    return sg_P-sg_K


# -----------------------------------------------------------------------------
# The sequel -- update the background distribution
# -----------------------------------------------------------------------------
num_it = 1;
end_base = 40.

for it in range(num_it):

    _, sg1, sg2 = result[0]

    sg1_instances = sg1.subgroupDescription.covers(data)
    sg1_nodes = list(np.where(sg1_instances)[0])

    sg2_instances = sg2.subgroupDescription.covers(data)
    sg2_nodes = list(np.where(sg2_instances)[0])

    # print(sg1_nodes)
    # print(sg2_nodes)

    (P_M, sg_K) = Connecting_Prob_BiCases(sg1_nodes, sg2_nodes, lambda_dict)

    sg1_N = len(sg1_nodes)
    sg2_N = len(sg2_nodes)

    if sg_K != 0:
        new_lambda = optimize.brentq(f_constraint, -end_base, end_base, args = (P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K))
        # new_lambda = optimize.newton(f_constraint, 10.)
        end_base += 20.
        # print(f_constraint(new_lambda))
    else:
        new_lambda = optimize.newton(f_constraint, 5., args = (P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K),maxiter=500)


    if new_lambda != 0.:
        lambda_dict[new_lambda] = [sg1_nodes,sg2_nodes]

    # print(lambda_dict)

    target = psx.GTarget_with_BlockCons(G, lambda_sum, lambda_dict, ud)

    # print(nx.adjacency_matrix(target.graph))
    # searchspace = psx.createSelectors(data,ignore=['high school','major','second major'])
    # searchspace = psx.createSelectors(data)

    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[6,8],depth=2,qf=psx.SubjectiveInterestingness())

    search_time_s = time.time()
    result = psx.BeamSearch(beamWidth=10).execute_in_bicases_constraints(task)
    search_time_e = time.time()

    search_time.append(search_time_e - search_time_s)

    Result.append(result)
    # print(lambda_dict)
    #
    # --------------------------
    Obj = (lambda_sum,lambda_dict,bg_time, search_time,Result)
    f = open(file,'wb')
    pickle.dump(Obj, f)
    f.close()
    # --------------------------

print(search_time)

for i in range(len(Result)):
    print('----------------------------------------')
    print('iteration' + str(i))
    print(Result[i])


# #
# #
# # -----------------------------------------------------------------------------
# # Save some key variables
# # -----------------------------------------------------------------------------
# cwd = os.getcwd()
# file = os.path.join(cwd, *['results','bi_' + name +'_results.pkl'])
#
# Obj = (x_rows, x_columns,rowbeans_index,colbeans_index,lambda_dict,bg_time, search_time,Result)
# f = open(file,'wb')
# pickle.dump(Obj, f)
# f.close()
#

# f = open('results/bi_lastfm_results.pkl', 'rb')
# f = open('results/bi_syn_1214_results.pkl', 'rb')
#
# f = open(file,'rb')
# Obj = pickle.load(f)
# f.close()
#
# for i in range(len(Obj[-1])):
#     print('----------------------------------------')
#     print('iteration' + str(i))
#     print(Obj[-1][i])

# #
# # -----------------------------------------------------------------------------
# # Visualization
# # -----------------------------------------------------------------------------
#
# SI, sg1, sg2 = Obj[-1][2][16]
#
# sg1_instances = sg1.subgroupDescription.covers(data)
# sg1_nodes = np.where(sg1_instances)[0]
#
# sg2_instances = sg2.subgroupDescription.covers(data)
# sg2_nodes = np.where(sg2_instances)[0]
#
# com_nodes = [i for i in sg1_nodes if i in sg2_nodes]
# Num_com = len(com_nodes)
#
#
# qf=psx.SubjectiveInterestingness()
#
# dict = sg1.target.lambda_dict
#
# # sg1.target.lambda_dict = {}
# # sg2.target.lambda_dict = {}
#
# sg1.target.lambda_dict = {k: dict[k] for k in list(dict)[:2]}
# sg2.target.lambda_dict = {k: dict[k] for k in list(dict)[:2]}
#
# (sg_K, sg_N, sg_P) = qf.computeStatistics_BiCases(data, sg1, sg2)
# num_attr = sg1.subgroupDescription.__len__() + \
#         sg2.subgroupDescription.__len__()
#
# SI_ = qf.computeSI(sg_K, sg_N, sg_P, num_attr)
# print(sg2.target.lambda_dict)
# print(SI_)
#
# title = 'SI = ' + '{:.03f}'.format(SI) + '      D1: ' + str(sg1.subgroupDescription) + \
# '      D2: ' + str(sg2.subgroupDescription)+ '      Num1: ' + str(len(sg1_nodes))+'      Num2: '\
#  + str(len(sg2_nodes))+ '       Num_com: ' + str(Num_com) + '      sg_K: ' + str(sg_K) +  \
#  '      sg_P = ' + '{:.03f}'.format(sg_P)
#
# cwd = os.getcwd()
# file = os.path.join(cwd, *['results','bi_' + name,'216.pdf'])
#
# if dataset == "s":
#     indicate_bicluster_in_smallNetwork(G, list(sg1_nodes), list(sg2_nodes), file, title)
# else:
#     indicate_bicluster_in_realNetwork(G, list(sg1_nodes), list(sg2_nodes), file, title)
