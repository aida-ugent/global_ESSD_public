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



# -----------------------------------------------------------------------------
# Get a graph and the user-attributes data frame from a particular dataset
# -----------------------------------------------------------------------------
# G, data, name = from_lastfm(11946)
# ud=True

G, data, name = from_dblp_affs()
ud=False

# adjacency matrix
total_A = nx.adjacency_matrix(G)


# -----------------------------------------------------------------------------
# Choosing Measurement
# -----------------------------------------------------------------------------

choice = input('IntraCluserDensity (Type 1) \n \
or Average Degree (Type 2) \n \
or InverseConductance (Type 3)\n \
or SimonMeasure (Type 4)\n \
or SegregationIndex (Type 5)\n \
or InverseAverageODF (Type 6)\n \
or LocalModularity (Type 7)\n \
or EdgeSurpls (Type 8)?')

# Initial background distribution
#
bd_graph = BGDistr(total_A, datasource='custom')
# x_rows, x_columns,rowbeans_index,colbeans_index,_,_ = bd_graph.compute_lambdas_in_a_cooler_way(iterations=1000,verbose=True,undirected=ud)
x_rows, x_columns,rowbeans_index,colbeans_index,_,_ = bd_graph.compute_lambdas_in_a_cooler_way(iterations=1000,verbose=True,undirected=ud,is_square_adj=False)
#
#
lambda_dict = {}
Result = []

target = psx.GTarget(G, x_rows, x_columns, rowbeans_index, colbeans_index, lambda_dict, ud)

searchspace = psx.createSelectors(data)
print(searchspace)
print(len(searchspace))

if choice == '1':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.IntraClusterDensity(),minQuality=-np.inf)
    measure = 'EdgeDensity'

elif choice == '2':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.AverageDegree(),minQuality=-np.inf)
    measure = 'AverageDegree'

elif choice == '3':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.InverseConductance(),minQuality=-np.inf)
    measure = 'InvConuctance'

elif choice == '4':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.SimonMeasure(),minQuality=-np.inf)
    measure = 'SimonM'

elif choice == '5':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.SegregationIndex(),minQuality=-np.inf)
    measure = 'SIDX'

elif choice == '6':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.InverseAverageODF(),minQuality=-np.inf)
    measure = 'IAODF'

elif choice == '7':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.LocalModularity(),minQuality=-np.inf)
    measure = 'MODL'

elif choice == '8':
    task = psx.SubgroupDiscoveryTask(data,target,searchspace,resultSetSize=[20],depth=2,qf=psx.graph_target.EdgeSurplus(),minQuality=-np.inf)
    measure = 'EdgeSurplus'

else:
    print("Invalid option.")
    measure = 'SI'

search_time = []
search_time_s = time.time()
result = psx.BeamSearch(beamWidth=30).execute(task)
search_time.append(time.time() - search_time_s)


Result.append(result)
print(result)

# save key variables
cwd = os.getcwd()
file = os.path.join(cwd, *['results','si_' + name + measure +'_results.pkl'])

Obj = (search_time,Result)
f = open(file,'wb')
pickle.dump(Obj, f)
f.close()
