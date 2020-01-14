
import numpy as np
import pandas as pd
import networkx as nx
import time

from scipy.sparse import csr_matrix

class GTarget_with_BlockCons(object):
    def __init__(self, graph, *args):
        self.graph = graph
        self.Adj_M = nx.adjacency_matrix(self.graph)
        self.lambda_sum = args[0]
        self.lambda_dict = args[1]
        self.undirected = args[2]

        # all the nodes
        Nodes = list(self.graph.nodes())
        N = len(self.graph)

        # a dictionary: nodes as keys and adjacency list as values
        # self.Adj_dict = {Nodes[k]: set(self.graph[Nodes[k]]) for k in range(N)}
        # self.Adj_dict = {k: self.Adj_M.indices[self.Adj_M.indptr[k]:self.Adj_M.indptr[k+1]] for k in range(N)}
        # self.edges = list(self.graph.edges())

    def __repr__(self):
        return "T: Density of connectivity"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)


    def get_base_statistics(self, data, subgroup, weightingAttribute=None):
        if (weightingAttribute is None):

            sg_instances = subgroup.subgroupDescription.covers(data)
            sg_nodes = list(np.where(sg_instances)[0])
            sg_n = np.sum(sg_instances)
            sg_N = sg_n*(sg_n-1.)

            sg_P = 0.

            if sg_N != 0:

                sg_A = nx.adjacency_matrix(self.graph,sg_nodes)
                sg_K = sg_A.count_nonzero()

                sg_lambda_sum = self.lambda_sum[sg_nodes][:,sg_nodes]
                P_M = np.exp(sg_lambda_sum)/(1. + np.exp(sg_lambda_sum))

                np.fill_diagonal(P_M, 0.)

                for i in self.lambda_dict.keys():

                    com_members = list(set(self.lambda_dict[i])&set(sg_nodes))
                    num_com_members = len(com_members)

                    if com_members != []:

                        com_members_ids = [sg_nodes.index(j) for j in com_members]

                        R = np.asarray(com_members_ids*num_com_members)
                        C = np.repeat(com_members_ids,num_com_members)

                        data = [i]*(num_com_members**2)

                        lambda_M = csr_matrix((data,(R,C)), shape=(sg_n, sg_n)).toarray()

                        new_odds_M = P_M * np.exp(lambda_M)

                        P_M = new_odds_M / (1. - P_M + new_odds_M)
                    else:
                        P_M = P_M
                #
                sg_P = np.sum(P_M)

                return (sg_K, sg_N, sg_P)


            else:
                return (0., 0., 1.)

        else:
            raise NotImplemented("Attribute weights with graph targets are not yet implemented.")



class SubjectiveInterestingness():
    def __init__(self):
            pass

    def computeSI(self, sg_K, sg_N, sg_P, num_attr):
        alpha = 0.3
        beta = 0.5

        if sg_N != 0.:

            if sg_N != sg_K:

                if sg_K != 0.:
                    IC = sg_K * np.log((sg_K - sg_K/sg_N*sg_P)/((1. - sg_K/sg_N)*sg_P)) + \
                    sg_N * np.log( (sg_N-sg_K) / (sg_N-sg_P))
                    #
                    # print('sg_K = %.4f' % sg_K)
                else:
                    IC = sg_N * (-np.log(1.- sg_P/sg_N))

            else:
                IC = sg_K * (-np.log(sg_P/sg_N))

            DI = alpha * num_attr + beta
            SI = IC / DI

        else:
            SI = 0.

        return SI

    def computeStatistics_BiCases(self, data, subgroup1, subgroup2, weightingAttribute=None):
        sg1_instances = subgroup1.subgroupDescription.covers(data)
        sg1_nodes = list(np.where(sg1_instances)[0])

        sg2_instances = subgroup2.subgroupDescription.covers(data)
        sg2_nodes = list(np.where(sg2_instances)[0])

        # print(sg1_nodes)
        # print(sg2_nodes)

        sg1_N = len(sg1_nodes)
        sg2_N = len(sg2_nodes)
        # print(sg1_N)
        # print(sg2_N)

        # finding the common members in sg1_nodes and sg2_nodes
        # print('haha')
        # time2=time.time()
        com_nodes = list(set(sg1_nodes)&set(sg2_nodes))
        # com_nodes = [i for i in sg1_nodes if i in sg2_nodes]
        # print(time.time()-time2)
        com_N = len(com_nodes)

        sg_N = sg1_N * sg2_N - com_N

        # print('sg_N = %.4f' % sg_N)

        # A = subgroup1.target.Adj_M

        if sg_N != 0:
            # time3=time.time()
            # ------------------------------------------------------------------
            # R = sg1_nodes*sg2_N
            # C = list(np.repeat(sg2_nodes,sg1_N))

            # The lower bound of the number of edges between sg1 and sg2
            # sg_K = A[R, C].sum()   # rows point to cols
            # # ------------------------------------------------------------------
            # adj_dict = subgroup1.target.Adj_dict
            # num_neighbors_in_sg2 = []
            # # sg2_nodes_set = OrderedSet(sg2_nodes)
            # sg2_nodes_set = set(sg2_nodes)
            # for i in sg1_nodes:
            #     num_neighbors_in_sg2.append(len(adj_dict[i]&sg2_nodes_set))
            # sg_K = sum(num_neighbors_in_sg2)
            # ------------------------------------------------------------------
            # union_nodes = list(set(sg1_nodes).union(set(sg2_nodes)))
            # H = subgroup1.target.graph.subgraph(union_nodes)
            # num_neighbors_in_sg2 = []
            # for i in sg1_nodes:
            #     num_neighbors_in_sg2.append(len(list(set(H[i])&set(sg2_nodes))))
            # sg_K = sum(num_neighbors_in_sg2)
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            R = np.asarray(sg1_nodes)
            C = np.asarray(sg2_nodes)
            # print(subgroup1.target.Adj_M[R[0],:])
            # print('here')
            # print(subgroup1.target.Adj_M[R[0:3],:])
            # print(subgroup1.target.Adj_M[:,C[0:3]])
            Rows = subgroup1.target.Adj_M[R]
            # The lower bound of the number of edges between sg1 and sg2
            sg_K = Rows[:,C].count_nonzero()
            # sg_K = np.sum(subgroup1.target.Adj_M[R[:,None],C])
            # sg_K = (subgroup1.target.Adj_M[R[:,None],C]==1).sum()
            # rows = subgroup1.target.Adj_M[R]
            # print(rows)
            # print(C)
            # sg_K = sum(row[0,C].count_nonzero() for row in rows)

            # sg_K = subgroup1.target.Adj_M[R[:,None],C].count_nonzero()
            # M = subgroup1.target.Adj_M.ravel()[(C + (R * subgroup1.target.Adj_M.shape[1]).reshape((-1,1))).ravel()]
            # sg_K = M.sum()
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # adj_dict = subgroup1.target.Adj_dict
            # sg_K = 0
            # for k in sg1_nodes:
            #     i = 0
            #     j = 0
            #     while i < len(adj_dict[k]) and j < sg2_N:
            #         if adj_dict[k][i] < sg2_nodes[j]:
            #             i += 1
            #         elif adj_dict[k][i] > sg2_nodes[j]:
            #             j += 1
            #         else:
            #             i += 1
            #             j += 1
            #             sg_K += 1
            # ------------------------------------------------------------------
            # print(time.time()-time3)
            # #
            # print('sg_K = %.4f' % sg_K)

            lambda_sum = subgroup1.target.lambda_sum[sg1_nodes][:,sg2_nodes]
            # #
            # # Add the block lambdas
            # #
            # for i in subgroup1.target.block_lambda_dict.keys():
            #
            #     com_members1 = list(set(subgroup1.target.block_lambda_dict[i][0])&set(sg1_nodes))
            #     com_members2 = list(set(subgroup1.target.block_lambda_dict[i][1])&set(sg2_nodes))
            #
            #     if com_members1 != [] and com_members2 != [] :
            #         # time4=time.time()
            #         com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members1])
            #         com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members2])
            #
            #         block_lambda_M = np.zeros((sg1_N, sg2_N))
            #         block_lambda_M[com_members_ids1[:,None],com_members_ids2]=i
            #
            #         lambda_sum = lambda_sum + block_lambda_M


            P_M = np.exp(lambda_sum)/(1. + np.exp(lambda_sum))

            # The probability of self-edge is 0
            for i in com_nodes:
                P_M[sg1_nodes.index(i), sg2_nodes.index(i)] = 0.


            # ------------------------------------------------------------------
            # Update the background distribution
            # ------------------------------------------------------------------
            for i in subgroup1.target.lambda_dict.keys():

                com_members1 = list(set(subgroup1.target.lambda_dict[i][0])&set(sg1_nodes))
                com_members2 = list(set(subgroup1.target.lambda_dict[i][1])&set(sg2_nodes))


                if com_members1 != [] and com_members2 != [] :
                    # time4=time.time()
                    com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members1])
                    com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members2])
                    # -----------------------------------------------------------------
                    # row = np.asarray(com_members_ids1 * num_com_members2)
                    # col = np.repeat(com_members_ids2, num_com_members1)
                    # value = [i]*(num_com_members1*num_com_members2)
                    # lambda_M = csr_matrix((value,(row,col)), shape=(sg1_N, sg2_N)).toarray()
                    # -----------------------------------------------------------------
                    lambda_M = np.ones((sg1_N, sg2_N))
                    lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(i)

                    odds_M = P_M * lambda_M
                    # print(time.time()-time4)
                else:
                    odds_M = P_M

                if subgroup1.target.undirected:

                    com_members1 = list(set(subgroup1.target.lambda_dict[i][1])&set(sg1_nodes))
                    com_members2 = list(set(subgroup1.target.lambda_dict[i][0])&set(sg2_nodes))


                    if com_members1 != [] and com_members2 != [] :

                        com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members1])
                        com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members2])
                        lambda_M = np.ones((sg1_N, sg2_N))
                        lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(i)

                        odds_M = odds_M * lambda_M

                P_M = odds_M / (1. - P_M + odds_M)
            # The expected number of edges between sg1 and sg2 under the
            # background distribution
            sg_P = np.sum(P_M)
            # print('sg_P = %.4f' % sg_P)

        else:
            sg_K = 0.
            sg_P = 1.

        return (sg_K, sg_N, sg_P)

    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        (sg_K, sg_N, sg_P) = subgroup.get_base_statistics(data, weightingAttribute)

        num_attr = subgroup.subgroupDescription.__len__()

        SI = self.computeSI(sg_K, sg_N, sg_P, num_attr)

        return SI

    def evaluateFromDataset_BiCases(self, data, subgroup1, subgroup2, weightingAttribute=None):
        (sg_K, sg_N, sg_P) = self.computeStatistics_BiCases(data, subgroup1, subgroup2, weightingAttribute)

        num_attr = subgroup1.subgroupDescription.__len__() + \
        subgroup2.subgroupDescription.__len__()

        SI = self.computeSI(sg_K, sg_N, sg_P, num_attr)

        return SI
