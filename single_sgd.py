import numpy as np
import pandas as pd
import networkx as nx
import pysubgroupx as psx

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy import optimize

from sklearn.preprocessing import MultiLabelBinarizer

from maxent.baseclass.optimization import *
from maxent.baseclass.MRDbase import *

from heapq import nlargest

import os

import time
import pickle

from visualisation import *
from get_graph_and_attributes import *

def f_Puv(x, overall_density):
    return np.exp(x+x)/(1+np.exp(x+x))-overall_density

def Connecting_Prob(sg_nodes,sg_x_rows,sg_x_cols, sg_n,lambda_dict):
    '''Computing the current connecting probability of edges within the subgroup'''
    sg_P = 0

    odds_M = sg_x_rows[:, None] + sg_x_cols
    P_M = np.exp(odds_M)/(1 + np.exp(odds_M))
    np.fill_diagonal(P_M, 0.)


    for i in lambda_dict.keys():

        com_members_ids = []

        com_members = list(set(lambda_dict[i])&set(sg_nodes))
        num_com_members = len(com_members)

        if com_members != []:

            com_members_ids = np.asarray([sg_nodes.index(j) for j in com_members])

            lambda_M = np.ones((sg_n, sg_n))
            lambda_M[com_members_ids[:,None],com_members_ids]=np.exp(i)

            odds_M = P_M * lambda_M
            P_M = odds_M / (1. - P_M + odds_M)

        else:

            P_M = P_M

    return P_M


def f_constraint(x, P_M, sg_n, sg_K):
    '''Computing the lambda'''
    update_lambda_M = np.exp(x)*np.ones((sg_n,sg_n))
    update_odds_M = P_M * update_lambda_M

    P_M = update_odds_M / (1. - P_M + update_odds_M)

    sg_P = np.sum(P_M)

    return sg_P-sg_K


# -----------------------------------------------------------------------------
# Get a graph and the user-attributes data frame from a particular dataset
# -----------------------------------------------------------------------------

dataset = input('Which dataset? \n \
The synthetic graph(Type: s), Delicious(Type d), Lastfm(Type l) \
Citeseer(Type c), Caltech36(Type ct), Reed98(Type r), DBLP topics as attributes(Type dbt)\
DBLP countries as attributes(Type dbc)')

if dataset == "s":
    G, data, name = from_old_synthesized_data()
    ud=False
elif dataset == "d":
    G, data, name = from_delicious(100)
    ud=True
elif dataset == "l":
    G, data, name = from_lastfm(11900)
    ud=True
elif dataset == "c":
    G, data, name = from_citeseer()
    ud=False
elif dataset == "ct":
    G, data, name = from_facebook100('Caltech36.mat')
    ud=True
elif dataset == "r":
    G, data, name = from_facebook100('Reed98.mat')
    ud=True
elif dataset == "dbt":
    G, data, name = from_dblp_topics()
    ud=False
elif dataset == "dbc":
    G, data, name = from_dblp_affs()
    ud=False
else:
    print("Invalid option.")

# adjacency matrix
adj_A = nx.adjacency_matrix(G)

# assert np.sum(np.abs(adj_A - adj_A.T)) != 0
# print('aaaaaaaaaaaaaaaaaaaaaaa', np.sum(np.abs(adj_A - adj_A.T)))

print(data.head())
print(data.columns)
print(adj_A.shape[0], np.sum(adj_A.todense()))

# -----------------------------------------------------------------------------
# Running choice
# -----------------------------------------------------------------------------

choice = input('Continue from a stored session (Type 1) \n \
or start a new search ? (Type 2)')

if choice == "1":
    which_prior = input('Pior beliefs on indiviual vertex degrees  (Type i) \n \
    or overall degree ? (Type o)')
    if which_prior == 'i':
        name = name + '_individual'
    elif which_prior == 'o':
        name = name + '_uniform'
    else:
        print('Invalid option')
    cwd = os.getcwd()
    file = os.path.join(cwd, *['results','si_' + name +'_results.pkl'])

    f = open(file,'rb')
    Obj = pickle.load(f)
    f.close()

    (x_rows, x_columns,rowbeans_index,colbeans_index,lambda_dict,bg_time, search_time, Result) = Obj
    result = Result[-1]
    print(result)

    target = psx.GTarget(G, x_rows, x_columns, rowbeans_index, colbeans_index, lambda_dict, ud)
    searchspace = psx.createSelectors(data)
    print(searchspace)
    print(len(searchspace))
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.SubjectiveInterestingness())


elif choice == "2":
    #
    # Initial background distribution
    #
    which_prior = input('Pior beliefs on indiviual vertex degrees  (Type i) \n \
    or overall degree ? (Type o)')

    if which_prior == 'i':
        name = name + '_individual'
        bd_graph = BGDistr(adj_A, datasource='custom')

        bg_time_s = time.time()
        x_rows, x_columns,rowbeans_index,colbeans_index,_,_ = bd_graph.compute_lambdas_in_a_cooler_way(iterations=1000,verbose=True,undirected=ud,is_square_adj=False)
        bg_time =  time.time() - bg_time_s

        lambda_dict = {}
        Result = []
        # -------------------------------------------------------------------------
        # Check weather probs are right
        all_nodes = list(range(len(G)))
        all_x_rows = x_rows[rowbeans_index[all_nodes]]
        all_x_cols = x_columns[colbeans_index[all_nodes]]

        lambda_sum = all_x_rows[:, None] + all_x_cols

        P_M = np.exp(lambda_sum)/(1. + np.exp(lambda_sum))

        # The probability of self-edge is 0
        np.fill_diagonal(P_M, 0.0)

        adjM = adj_A.todense()

        diff_row = np.sum(P_M,axis=0).squeeze()-np.sum(adjM, axis=0).squeeze()
        diff_col = np.sum(P_M,axis=1).squeeze()-np.sum(adjM, axis=1).squeeze()

        print(diff_row[:,:20])
        print(diff_col[:,:20])
        # -------------------------------------------------------------------------

    elif which_prior == 'o':
        name = name +'_uniform'
        overall_density = adj_A.count_nonzero()/(adj_A.shape[0]*(adj_A.shape[0]-1))
        bg_time_s = time.time()
        lamb = optimize.fsolve(f_Puv, 1., args = (overall_density))
        bg_time =  time.time() - bg_time_s

        x_rows = np.asarray(lamb)
        x_columns = np.asarray(lamb)
        rowbeans_index = np.array([0]*(adj_A.shape[0]))
        colbeans_index = np.array([0]*(adj_A.shape[0]))

        lambda_dict = {}
        Result = []

    else:
        print('Invalid option')

    # The first search
    #
    target = psx.GTarget(G, x_rows, x_columns, rowbeans_index, colbeans_index, lambda_dict, ud)

    searchspace = psx.createSelectors(data)
    print(searchspace)
    print(len(searchspace))

    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.SubjectiveInterestingness())
    # task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[8],depth=3,qf=psx.graph_target.IntraClusterDensity())
    # task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[8],depth=3,qf=psx.graph_target.InverseConductance())
    # task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[8],depth=3,qf=psx.graph_target.SimonMeasure())
    # task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[8],depth=3,qf=psx.graph_target.SegregationIndex())

    search_time = []
    search_time_s = time.time()
    result = psx.BeamSearch(beamWidth=30).execute(task)
    search_time.append(time.time() - search_time_s)

    Result.append(result)
    print(result)

    # save key variables
    cwd = os.getcwd()
    file = os.path.join(cwd, *['results','si_' + name +'_results.pkl'])

    Obj = (x_rows, x_columns,rowbeans_index,colbeans_index,lambda_dict,bg_time, search_time,Result)
    f = open(file,'wb')
    pickle.dump(Obj, f)
    f.close()

else:
    print("Invalid option.")



# -----------------------------------------------------------------------------
# The sequel
# -----------------------------------------------------------------------------

num_it = 5;
end_base = 50.

for it in range(num_it):

    _, sg = result[0]

    sg_nodes, sg_x_rows, sg_x_cols = sg.get_lambdas(data, weightingAttribute=None)
    sg_n = len(sg_nodes)

    P_M = Connecting_Prob(sg_nodes, sg_x_rows, sg_x_cols, sg_n, lambda_dict)

    Ids = np.asarray(sg_nodes)
    sg_K = target.Adj_M[Ids][:,Ids].count_nonzero()

    if sg_K != 0:
        new_lambda = optimize.brentq(f_constraint, -end_base, end_base, args = (P_M, sg_n, sg_K))
        end_base += 20.
    else:
        new_lambda = optimize.newton(f_constraint, 5., args = (P_M, sg_n, sg_K))

    if new_lambda != 0.:
        lambda_dict[new_lambda] = sg_nodes

    target = psx.GTarget(G,x_rows, x_columns, rowbeans_index, colbeans_index, lambda_dict,ud)

    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[10],depth=2,qf=psx.graph_target.SubjectiveInterestingness())

    result = psx.BeamSearch(beamWidth=30).execute(task)
    Result.append(result)

    print(lambda_dict)

    # --------------------------
    Obj = (x_rows, x_columns,rowbeans_index,colbeans_index,lambda_dict,bg_time, search_time,Result)
    f = open(file,'wb')
    pickle.dump(Obj, f)
    f.close()
    # --------------------------

for i in range(len(Result)):
    print('----------------------------------------')
    print('iteration' + str(i))
    print(Result[i])
#
# # -----------------------------------------------------------------------------
# # Visualization
# # -----------------------------------------------------------------------------
#
# SI, sg = Obj[-1][0][0]
#
# sg_instances = sg.subgroupDescription.covers(data)
# sg_nodes = np.where(sg_instances)[0]
# (sg_K, sg_N, sg_P) = sg.get_base_statistics(data, weightingAttribute=None)
#
# title = 'SI = ' + '{:.03f}'.format(SI) + '      D: ' + str(sg.subgroupDescription) + '      Num: ' + str(len(sg_nodes))\
# + '      sg_K: ' + str(sg_K) +  '      sg_P = ' + '{:.03f}'.format(sg_P)
#
# indicate_cluster_in_realNetwork_u(G, list(sg_nodes), 'results/lastfm1.pdf', title)
