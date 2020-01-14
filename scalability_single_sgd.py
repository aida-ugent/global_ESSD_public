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


from get_graph_and_attributes import *


# testing_list = [5,10,20]
testing_list = [5,10,20,40,80,160,320,640,1280,2560,5120,10240]
search_time = []

for i in testing_list:

    G, data, name = from_lastfm(i)
    ud = True

    total_A = nx.adjacency_matrix(G)

    # Initial background distribution
    #
    bd_graph = BGDistr(total_A, datasource='custom')
    x_rows, x_columns,rowbeans_index,colbeans_index,_,_ = bd_graph.compute_lambdas_in_a_cooler_way(iterations=1000,verbose=True,undirected=ud)
    lambda_dict = {}
    Result = []

    # # -------------------------------------------------------------------------
    # # Check weather probs are right
    # all_nodes = list(range(len(G)))
    # all_x_rows = x_rows[rowbeans_index[all_nodes]]
    # all_x_cols = x_columns[colbeans_index[all_nodes]]
    #
    # lambda_sum = all_x_rows[:, None] + all_x_cols
    #
    # P_M = np.exp(lambda_sum)/(1. + np.exp(lambda_sum))
    #
    # # The probability of self-edge is 0
    # np.fill_diagonal(P_M, 0.0)
    #
    # adjM = total_A.todense()
    #
    # diff_row = np.sum(P_M,axis=0).squeeze()-np.sum(adjM, axis=0).squeeze()
    # diff_col = np.sum(P_M,axis=1).squeeze()-np.sum(adjM, axis=1).squeeze()
    #
    # print(diff_row[:,:20])
    # print(diff_col[:,:20])

    #
    # The first search
    #
    target = psx.GTarget(G, x_rows, x_columns, rowbeans_index, colbeans_index, lambda_dict, ud)

    searchspace = psx.createSelectors(data)
    print(len(searchspace))

    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.SubjectiveInterestingness())
    search_time_s = time.time()
    result = psx.BeamSearch(beamWidth=30).execute(task)
    search_time.append(time.time() - search_time_s)

    Result.append(result)

# save key variables
cwd = os.getcwd()
file = os.path.join(cwd, *['results','scalability_' + name +'_varyingNumAttr.pkl'])

Obj = (testing_list,search_time)
f = open(file,'wb')
pickle.dump(Obj, f)
f.close()

f = open(file,'rb')
Obj = pickle.load(f)
f.close()
print(Obj)
