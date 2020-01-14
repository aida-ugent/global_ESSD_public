import numpy as np
import pandas as pd
import networkx as nx
import time
import math
from scipy.sparse import csr_matrix
from scipy import optimize

class NwTarget(object):
    def __init__(self, graph, *args):
        self.graph = graph
        self.Adj_M = nx.adjacency_matrix(self.graph)
        self.x_rows = args[0]
        self.x_columns = args[1]
        self.rowbeans_index = args[2]
        self.colbeans_index = args[3]
        self.end_base = args[4]
        # self.undirected = args[4]

        # all the nodes
        Nodes = list(self.graph.nodes())
        N = len(self.graph)

    def get_original_lambdas(self, sg_nodes):

        sg_x_rows = self.x_rows[self.rowbeans_index[sg_nodes]]
        sg_x_cols = self.x_columns[self.colbeans_index[sg_nodes]]
        # print('sg_x_rows', sg_x_rows)
        # print('sg_x_cols', sg_x_cols)

        return sg_x_rows, sg_x_cols

    def f_constraint_SingleGroup(self, x, P_M, sg_n, sg_K):
        '''Computing the lambda'''
        update_lambda_M = np.exp(x)*np.ones((sg_n,sg_n))
        update_odds_M = P_M * update_lambda_M

        update_P_M = update_odds_M / (1. - P_M + update_odds_M)

        sg_P = np.sum(update_P_M)

        return sg_P-sg_K

    def f_constraint_BiGroups(self, x, P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K):
        '''Computing the lambda'''
        lambda_M = np.exp(x)*np.ones((sg1_N,sg2_N))
        update_odds_M = P_M * lambda_M

        # if self.undirected:
        #     com_members = list(set(sg1_nodes)&set(sg2_nodes))
        #
        #     if com_members != [] :
        #
        #         com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members])
        #         com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members])
        #
        #         lambda_M = np.ones((sg1_N, sg2_N))
        #         lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(x)
        #
        #         update_odds_M = update_odds_M * lambda_M

        update_P_M = update_odds_M / (1. - P_M + update_odds_M)

        sg_P = np.sum(update_P_M)

        return sg_P-sg_K

    def get_statistics_SingleGroup(self, sg_nodes, lambda_dict):
        sg_n = len(sg_nodes)

        Ids = np.asarray(sg_nodes)

        sg_Adj_M = self.Adj_M[Ids][:,Ids]
        sg_K = sg_Adj_M.count_nonzero()

        sg_x_rows, sg_x_cols = self.get_original_lambdas(sg_nodes)

        lambda_sum = sg_x_rows[:,None] + sg_x_cols
        P_M = np.exp(lambda_sum)/(1. + np.exp(lambda_sum))

        np.fill_diagonal(P_M, 0.)

        for i in lambda_dict.keys():

            com_members = list(set(lambda_dict[i][0])&set(sg_nodes))
            num_com_members = len(com_members)

            if com_members != []:

                com_members_ids = np.asarray([sg_nodes.index(j) for j in com_members])

                lambda_M = np.ones((sg_n, sg_n))
                lambda_M[com_members_ids[:,None],com_members_ids]=np.exp(i)

                odds_M = P_M * lambda_M
                P_M = odds_M / (1. - P_M + odds_M)
            else:
                P_M = P_M


        if sg_K != 0 and self.f_constraint_SingleGroup(-self.end_base, P_M, sg_n, sg_K)*self.f_constraint_SingleGroup(self.end_base,P_M, sg_n, sg_K) < 0:
            new_lambda = optimize.brentq(self.f_constraint_SingleGroup, -self.end_base, self.end_base, args = (P_M, sg_n, sg_K))
            self.end_base += 3.
        else:
            # new_lambda = optimize.fixed_point(self.f_constraint_SingleGroup, 0., args = (P_M, sg_n, sg_K))
            # new_lambda = optimize.newton(self.f_constraint_SingleGroup, 2., args = (P_M, sg_n, sg_K),tol = 1.0, maxiter=5000)
            new_lambda = optimize.fsolve(self.f_constraint_SingleGroup, 1., args = (P_M, sg_n, sg_K))



        # compute the updated P_M if we temporarily coorporate this block
        temp_lambda_M = np.exp(new_lambda)*np.ones((sg_n,sg_n))
        temp_odds_M = P_M * temp_lambda_M

        temp_P_M = temp_odds_M / (1. - P_M + temp_odds_M)

        # compute log likelihood
        sg_Adj_M = sg_Adj_M.toarray()
        likelihood = np.multiply(P_M, sg_Adj_M)+np.multiply(1-P_M, 1-sg_Adj_M)
        logLike = np.sum(np.log(likelihood))

        temp_likelihood = np.multiply(temp_P_M, sg_Adj_M)+np.multiply(1-temp_P_M, 1-sg_Adj_M)
        temp_logLike = np.sum(np.log(temp_likelihood))

        IC_block = temp_logLike - logLike


        return new_lambda, IC_block



    def get_statistics_BiGroups(self,  sg1_nodes, sg2_nodes, lambda_dict):
        sg1_x_rows, sg1_x_cols = self.get_original_lambdas(sg1_nodes)
        sg2_x_rows, sg2_x_cols = self.get_original_lambdas(sg2_nodes)

        sg1_N = len(sg1_nodes)
        sg2_N = len(sg2_nodes)

        com_nodes = list(set(sg1_nodes)&set(sg2_nodes))

        com_N = len(com_nodes)


        R = np.asarray(sg1_nodes)
        C = np.asarray(sg2_nodes)

        Rows = self.Adj_M[R]
        sg_Adj_M = Rows[:,C]

        sg_K = sg_Adj_M.count_nonzero()

        lambda_sum = sg1_x_rows[:, None] + sg2_x_cols
        P_M = np.exp(lambda_sum)/(1. + np.exp(lambda_sum))

        # The probability of self-edge is 0
        for i in com_nodes:
            P_M[sg1_nodes.index(i), sg2_nodes.index(i)] = 0.

        # ------------------------------------------------------------------
        # update the background distribution
        # ------------------------------------------------------------------
        for i in lambda_dict.keys():

            com_members1 = list(set(lambda_dict[i][0])&set(sg1_nodes))
            com_members2 = list(set(lambda_dict[i][1])&set(sg2_nodes))

            if com_members1 != [] and com_members2 != [] :
                # time4=time.time()
                com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members1])
                com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members2])

                lambda_M = np.ones((sg1_N, sg2_N))
                lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(i)

                odds_M = P_M * lambda_M
                # print(time.time()-time4)
            else:
                odds_M = P_M

            # if self.undirected:
            #
            #     com_members1 = list(set(lambda_dict[i][1])&set(sg1_nodes))
            #     com_members2 = list(set(lambda_dict[i][0])&set(sg2_nodes))
            #
            #
            #     if com_members1 != [] and com_members2 != [] :
            #
            #         com_members_ids1 = np.asarray([sg1_nodes.index(j) for j in com_members1])
            #         com_members_ids2 = np.asarray([sg2_nodes.index(j) for j in com_members2])
            #         lambda_M = np.ones((sg1_N, sg2_N))
            #         lambda_M[com_members_ids1[:,None],com_members_ids2]=np.exp(i)
            #
            #         odds_M = odds_M * lambda_M

            P_M = odds_M / (1. - P_M + odds_M)


        if sg_K != 0 and self.f_constraint_BiGroups(-self.end_base,P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K)*self.f_constraint_BiGroups(self.end_base,P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K) < 0:
            new_lambda = optimize.brentq(self.f_constraint_BiGroups, -self.end_base, self.end_base, args = (P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K))
            # new_lambda = optimize.newton(f_constraint, 10.)
            self.end_base += 3.
            # print(f_constraint(new_lambda))
        else:
            # new_lambda = optimize.fixed_point(self.f_constraint_BiGroups, 0., args = (P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K), xtol=1.5, maxiter=5000)
            new_lambda = optimize.fsolve(self.f_constraint_BiGroups, 1., args = (P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K))
            # new_lambda = optimize.newton(self.f_constraint_BiGroups, 2., args = (P_M, sg1_nodes, sg2_nodes, sg1_N, sg2_N, sg_K),tol = 1.5,maxiter=5000)

        # print(new_lambda)

        # compute the updated P_M if we temporarily coorporate this block
        temp_lambda_M = np.exp(new_lambda)*np.ones((sg1_N,sg2_N))
        temp_odds_M = P_M * temp_lambda_M

        temp_P_M = temp_odds_M / (1. - P_M + temp_odds_M)


        # compute log likelihood
        sg_Adj_M = sg_Adj_M.toarray()
        likelihood = np.multiply(P_M, sg_Adj_M)+np.multiply(1-P_M, 1-sg_Adj_M)
        logLike = np.sum(np.log(likelihood))
        # print(logLike)

        temp_likelihood = np.multiply(temp_P_M, sg_Adj_M)+np.multiply(1-temp_P_M, 1-sg_Adj_M)
        temp_logLike = np.sum(np.log(temp_likelihood))
        # print(temp_logLike)

        IC_block = temp_logLike - logLike


        return new_lambda, IC_block
