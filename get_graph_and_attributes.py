import numpy as np
import pandas as pd
import networkx as nx
import pysubgroupx as psx
import scipy
import warnings

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy import optimize

from sklearn.preprocessing import MultiLabelBinarizer

from maxent.baseclass.optimization import *
from maxent.baseclass.MRDbase import *

from load_dblp_data import (load_dblp_data, load_dblp_data_affs, compute_adj, compute_attributes,
                            compute_LSA)
from utils import memoize, mkdir

from heapq import nlargest

import os


def from_old_synthesized_data():
    indptr=np.array([0,4,7,9,11,12,13,16,18,19,19])
    indices=np.array([1,2,3,6,2,3,4,3,5,4,5,5,8,7,8,9,8,9,9])
    data=np.ones(19)

    A = csr_matrix((data, indices, indptr), dtype=np.int8,shape=(10,10))
    print(A)

    G = nx.from_scipy_sparse_matrix(A)
    print(nx.adjacency_matrix(G).todense())
    # G.remove_edges_from(G.selfloop_edges())

    # adding attributes
    d = {'a': [1,0,1,0,1,0,0,1,0,0], 'b': [0,0,1,0,0,1,1,1,1,1], 'c': [1,1,0,1,0,1,0,0,0,0], \
    'd': [0,0,0,0,1,0,0,0,1,0], 'e': [0,1,0,0,0,0,1,1,1,1]}
    data = pd.DataFrame(data=d)

    return G, data, 'old_syn'


def from_new_synthesized_data():
    indptr=np.array([0,4,6,8,8,10,11,14,16,18,18,18])
    indices=np.array([1,2,3,4,2,3,3,9,5,7,6,8,9,10,8,10,9,10])
    data=np.ones(18)

    A = csr_matrix((data, indices, indptr), dtype=np.int8,shape=(11,11))
    print(A.todense())

    G = nx.from_scipy_sparse_matrix(A)
    # G.remove_edges_from(G.selfloop_edges())

    # d = {'a': [0,0,0,0,1,0,0,1,0,1,1], 'b': [1,1,1,1,1,0,0,1,0,0,0], 'c': [0,0,1,0,0,0,1,1,1,1,1], \
    # 'd': [1,0,1,1,1,0,0,0,0,1,0], 'e': [1,1,1,0,1,0,1,0,0,1,1]}

    # adding attributes
    d = {'a': [0,0,0,0,1,0,0,1,0,1,1], 'b': [1,1,1,1,1,0,0,1,0,0,0], 'c': [0,0,1,0,0,0,1,1,1,1,1], \
    'd': [1,0,1,1,1,0,0,0,0,1,0]}

    data = pd.DataFrame(data=d)
    return G, data, 'new_syn'


def from_synthesized_bi_data():
    ## sythesize from csr matrix
    #
    # indptr=np.array([0,2,5,5,9,13,15,15,15,15,16])
    # indices=np.array([2,4,0,2,3,6,7,8,9,6,7,8,9,7,8,5])
    # data=np.ones(16)
    #
    # A = csr_matrix((data, indices, indptr), dtype=np.int8,shape=(10,10))
    #
    # G = nx.from_scipy_sparse_matrix(A,create_using=nx.DiGraph)

    ## synthesize a graph from numpy matrix
    adj = np.zeros((10,10))
    adj[0,1:3]=1
    adj[1,0]=1
    adj[1,2]=1
    adj[2,0:2]=1
    adj[3:6,6:10]=1
    adj[6,0:2]=1
    adj[7,0]=1
    adj[7,2]=1
    adj[8,1:3]=1
    adj[9,2]=1
    # adj[9,0]=1

    G = nx.from_numpy_matrix(adj,create_using=nx.DiGraph)
    A = nx.adjacency_matrix(G)

    print(A)

    # adding attributes
    d = {'a': [0,0,0,1,1,1,1,1,0,0], 'b': [1,1,1,0,0,0,1,1,0,1],'c': [0,1,0,0,1,0,1,1,1,1],\
     'd': [1,1,0,1,1,0,0,0,0,0]}
    data = pd.DataFrame(data=d)
    # print(data)
    # G.remove_edges_from(G.selfloop_edges())

    return G, data, 'syn'


def from_facebook100(name):
    cwd = os.getcwd()
    file = os.path.join(cwd, *['datasets','facebook100', name])

    Dataset = scipy.io.loadmat(file)

    A = Dataset['A']
    data = Dataset['local_info']

    G = nx.from_scipy_sparse_matrix(A)
    total_A = nx.adjacency_matrix(G)
    # print(total_A)
    # print(G.nodes)
    # print(G.edges)

    # sg_A = nx.adjacency_matrix(G,[0,4])
    # print(sg_A.count_nonzero())
    # print(nx.adjacency_matrix(G))

    data = pd.DataFrame.from_records(Dataset['local_info'])
    # data.columns = [0,1,2,3,4,5,6]
    data = data.astype(str)
    data.columns=['student/faculty status flag', 'gender', 'major', 'second major', \
    'dorm/house', 'year', 'high school']
    # print(data)
    # data = data.apply(pd.to_numeric)
    # print(data.head())
    # print(data.dtypes)
    # print(data.columns)

    return G, data, name[:-4]

def from_delicious(num_attr):
    cwd = os.getcwd()
    file_re = os.path.join(cwd, *['datasets','Delicious', 'user_contacts.dat'])
    file_tags = os.path.join(cwd, *['datasets','Delicious', 'tags.dat'])

    file_attr = os.path.join(cwd, *['datasets','Delicious', 'user_taggedbookmarks.dat'])

    # Read the tag dataset
    data_tag = pd.read_csv(file_tags,
                        sep='\t',
                        encoding="latin1")


    tagID = list(data_tag[data_tag.columns[0]])
    tagValue = list(data_tag[data_tag.columns[-1]])

    print(tagID[0])

    tag_mapping = dict(zip(tagID,tagValue))

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

    print(data_re[0:6])

    G=nx.Graph()

    G.add_nodes_from(np.unique(userID))
    G.add_edges_from(data_re)

    N = len(G)
    print(N)
    # print(list(G.nodes)[:10])
    # print(nx.adjacency_matrix(G))


    node_mapping=dict(zip(G.nodes(),range(N)))
    # nx.relabel_nodes(G,node_mapping,copy=False)

    # print(list(G.nodes)[:10])
    # print(node_mapping)

    print(list(G.edges)[:10])
    # print(nx.adjacency_matrix(G))
    print(nx.adjacency_matrix(G)[0])

    # Construct the user-attributes data frame
    data_attr = np.genfromtxt(file_attr,
                         names=True,
                         dtype=None,
                         usecols=(0, 2))

    # a dicitonary with the keys as the users and values as the tags each user assigned
    attr_dict = {node_mapping[k[0]]: set() for k in data_attr}

    # a dictionary with the keys as the tags and values as the assignment frequency
    # of each tag
    attr_freq_dict = dict.fromkeys(tagValue,0)



    ## get the 1350 most frequent tagsprior
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

    data = pd.DataFrame.from_dict(attr_dict, orient='index',columns=attrs)

    print(data.head())
    print(data.columns)

    return G, data, 'delicious'


def from_lastfm(num_attr):
    cwd = os.getcwd()
    file_re = os.path.join(cwd, *['datasets','Lastfm', 'user_friends.dat'])
    file_tags = os.path.join(cwd, *['datasets','Lastfm', 'tags.dat'])

    file_attr = os.path.join(cwd, *['datasets','Lastfm', 'user_taggedartists.dat'])

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

    N = len(G)
    print(N)
    # print(list(G.nodes)[:10])
    # print(nx.adjacency_matrix(G))

    node_mapping=dict(zip(G.nodes(),range(N)))
    # nx.relabel_nodes(G,node_mapping,copy=False)

    # print(list(G.nodes)[:10])
    # print(node_mapping)

    print(list(G.edges)[:10])
    print(len(list(G.edges)))
    # print(nx.adjacency_matrix(G))
    print(nx.adjacency_matrix(G)[0:2])
    print(nx.adjacency_matrix(G)[0][:,np.array((257,400,482))])
    print(nx.adjacency_matrix(G)[np.array((0,1,2))][:,np.array((257,400,482))].count_nonzero())
    print(nx.adjacency_matrix(G)[257])


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
    ## get the 1350 most frequent tagsprior
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

    data = pd.DataFrame.from_dict(attr_dict, orient='index',columns=attrs)

    print(data.index)
    # print(data.head())
    print(data.columns)

    return G, data, 'lastfm'


def from_citeseer():
    cwd = os.getcwd()
    file_re = os.path.join(cwd, *['datasets','citeseer', 'citeseer.cites'])
    file_attr = os.path.join(cwd, *['datasets','citeseer', 'citeseer.content'])

    # Read the paper content data
    data_attr = pd.read_csv(file_attr,
                        dtype=None,
                        header=None,
                        sep='\t')

    print(data_attr.head())

    data_attr[0] = data_attr[0].astype(str)
    paper_ids = list(data_attr[0].values)
    print(paper_ids[:10])


    # Construct a grpah correspoinding to the friend relationship dataset
    # Note in this data, if a line is represented by "paper1 paper2" then the link is "paper2->paper1"
    data_re = np.genfromtxt(file_re,
                         # names=True,
                         dtype=None,
                         encoding="latin1",
                         usecols=(1,0))

    print(data_re[1])

    G=nx.DiGraph()

    G.add_nodes_from(paper_ids)
    G.add_edges_from(data_re)

    N = len(G)
    print(N)

    mapping=dict(zip(G.nodes(),range(N)))
    nx.relabel_nodes(G,mapping,copy=False)

    # Construct the user-attributes data frame
    data_attr = pd.read_csv(file_attr,
                        dtype=None,
                        header=None,
                        sep='\t')

    # print(data_attr.head())

    data_attr[0] = data_attr[0].astype(str)

    paperids = []
    for i in list(data_attr[0].values):
        paperids.append(mapping[i])

    print(paperids[:10])

    data = data_attr.select_dtypes(include=['number'])
    data.index = paperids

    print(data.head())

    return G, data, 'citeseer'


def from_germany():
    cwd = os.getcwd()
    file = os.path.join(cwd, *['datasets','germany', 'socio_economics_germany_2009_v3.csv'])

    total_data = pd.read_csv(file,
                    sep=';',
                    encoding="latin1")


    # Construct a graph correspoinding to the friend relationship dataset
    # the nodes
    AreaCode = list(total_data['Area Code'])
    AreaName = list(total_data['Area Name'])

    N = len(AreaCode)
    AreaMapping = dict(zip(AreaCode, AreaName))

    # the edges
    edgelist_dict = {k : [] for k in AreaCode}

    # print(total_data.at[411,'Neighborhood'])

    for i in range(N):
        neighbors = total_data.at[i ,'Neighborhood']

        for j in neighbors.split(','):

            edgelist_dict[AreaCode[i]].append(int(j))   # convert the string element to int


    print(edgelist_dict[1001])
    # the graph
    G=nx.from_dict_of_lists(edgelist_dict)

    # relabel the nodes by ordered numbers
    node_mapping = dict(zip(G.nodes(),range(N)))


    nx.relabel_nodes(G,node_mapping,copy=False)

    # for i in range(N):
    #     print(AreaName[node_mapping[AreaCode[0]]]==AreaMapping[AreaCode[0]])


    print(node_mapping[1051])

    total_A = nx.adjacency_matrix(G)
    M = total_A.count_nonzero()
    print(M)
    # print(total_A.todense())


    # Construct the user-attributes data frame
    data = total_data.iloc[:,7:45]
    data =data.replace(',','.', regex=True).astype(np.float16)
    data.index = range(N)

    print(data.head())

    return G, data, 'germany'

# dblp citation network with research topics as attributes
def from_dblp_topics():
    # supress the scipy warning:
    # RuntimeWarning: The number of calls to function has reached maxfev = 400.
    warnings.simplefilter("ignore")

    data_name = 'dblp_four_areas'
    data_folder = './datasets/'
    cache_folder = mkdir('./cache/')
    # summary_file = 'dblp_summary.pkl'
    # max_num_selector = 1

    # load dblp_data
    raw_data_file = os.path.join(data_folder, 'dblp_papers_v11.txt')
    paper_records_cache_file = os.path.join(cache_folder, 'dblp_paper_records.pkl')
    dblp_data_cache_file = os.path.join(cache_folder, 'dblp_data.pkl')
    adj_A_cache_file = os.path.join(cache_folder, 'dblp_adj_A.pkl')
    att_A_cache_file = os.path.join(cache_folder, 'dblp_att_A.pkl')

    subsample_step = 8
    dblp_data = memoize(load_dblp_data, dblp_data_cache_file)(data_name,
                            raw_data_file, paper_records_cache_file,
                            subsample_step)
    adj_A = memoize(compute_adj, adj_A_cache_file, refresh=False)(dblp_data)
    att_A = memoize(compute_attributes, att_A_cache_file,
                    refresh=False)(dblp_data)

    adj_A = adj_A[::1, ::1]
    print(adj_A.shape)

    G = nx.from_scipy_sparse_matrix(adj_A, create_using=nx.DiGraph())
    att_A = att_A[::1]

    # perform LSA, center the data
    LSA_cache_file = os.path.join(cache_folder, 'dblp_lsa.pkl')
    att_A -= np.mean(att_A, axis=0)
    lsi_A = memoize(compute_LSA, LSA_cache_file, refresh=False)(att_A, 50)

    attr_proj = att_A.dot(lsi_A)

    data = pd.DataFrame(attr_proj)
    data.columns = list(map(str, list(range(50))))

    return G, data, 'dblp_topics'

# dblp citation network with countries as attributes
def from_dblp_affs():
    # supress the scipy warning:
    # RuntimeWarning: The number of calls to function has reached maxfev = 400.
    warnings.simplefilter("ignore")

    data_name = 'dblp_four_areas'
    data_folder = './datasets/'
    cache_folder = mkdir('./cache/')

    raw_data_file = os.path.join(data_folder, 'dblp_papers_v11.txt')
    paper_records_cache_file = os.path.join(cache_folder, 'dblp_paper_records_affs.pkl')
    dblp_data_cache_file = os.path.join(cache_folder, 'dblp_data_affs.pkl')
    adj_A_cache_file = os.path.join(cache_folder, 'dblp_adj_A_affs.pkl')
    att_A_cache_file = os.path.join(cache_folder, 'dblp_att_A_affs.pkl')
    subsample_step = 8

    dblp_data, data = memoize(load_dblp_data_affs, dblp_data_cache_file,refresh=True)(data_name,
                            raw_data_file, paper_records_cache_file,
                            subsample_step)
    adj_A = memoize(compute_adj, adj_A_cache_file, refresh=True)(dblp_data)
    adj_A = adj_A[::1, ::1]

    G = nx.from_scipy_sparse_matrix(adj_A, create_using=nx.DiGraph())

    return G, data, 'dblp_affs'
