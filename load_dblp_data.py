import json
import os
import pickle
import random
import sys
from collections import defaultdict
from os.path import join

import numpy as np
import pandas as pd
import itertools

import scipy.sparse as sparse
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import *

from utils import (mkdir, memoize)

import pycountry


def load_venues(data_name):
    if data_name == 'dblp_ml_dm':
        return [
            {'raw': 'knowledge discovery and data mining', 'id': '1130985203', 'acronym': 'KDD'},
            {'raw': 'european conference on principles of data mining and knowledge discovery', 'id': '1141769385', 'acronym': 'ECML-PKDD'},
            {'raw': 'Data Mining and Knowledge Discovery', 'id': '121920818', 'acronym': 'DAMI'},
            {'raw': 'neural information processing systems', 'id': '1127325140', 'acronym': 'NIPS'},
            {'raw': 'Journal of Machine Learning Research', 'id': '118988714', 'acronym': 'JMLR'},
            {'raw': 'international conference on machine learning', 'id': '1180662882', 'acronym': 'ICML'},
            {'raw': 'Machine Learning', 'id': '62148650', 'acronym': 'MLJ'},
            {'raw': 'international conference on learning representations', 'id': '2584161585', 'acronym': 'ICLR'}]
    elif data_name == 'dblp_four_areas':
        return [
            {'raw': 'international joint conference on artificial intelligence', 'id': '1203999783', 'acronym': 'IJCAI', 'area': 'ML'},
            {'raw': 'national conference on artificial intelligence', 'id': '1184914352', 'acronym': 'AAAI', 'area': 'ML'},
            {'raw': 'international conference on machine learning', 'id': '1180662882', 'acronym': 'ICML', 'area': 'ML'},
            {'raw': 'neural information processing systems', 'id': '1127325140', 'acronym': 'NIPS', 'area': 'ML'},
            {'raw': 'international conference on learning representations', 'id': '2584161585', 'acronym': 'ICLR', 'area': 'ML'},
            {'raw': 'international conference on data engineering', 'id': '1163988186', 'acronym': 'ICDE', 'area': 'DB'},
            {'raw': 'very large data bases', 'id': '1133523790', 'acronym': 'VLDB', 'area': 'DB'},
            {'raw': 'international conference on management of data', 'id': '1175089206', 'acronym': 'SIGMOD', 'area': 'DB'},
            {'raw': 'international conference on database theory', 'id': '1165285842', 'acronym': 'ICDT', 'area': 'DB'},
            {'raw': 'symposium on principles of database systems', 'id': '1184151122', 'acronym': 'PODS', 'area': 'DB'},
            {'raw': 'international acm sigir conference on research and development in information retrieval', 'id': '1140684652', 'acronym': 'SIGIR', 'area': 'IR'},
            {'raw': 'international world wide web conferences', 'id': '1135342153', 'acronym': 'WWW', 'area': 'IR'},
            {'raw': 'conference on information and knowledge management', 'id': '1194094125', 'acronym': 'CIKM', 'area': 'IR'},
            {'raw': 'european conference on information retrieval', 'id': '1180513217', 'acronym': 'ECIR', 'area': 'IR'},
            {'raw': 'knowledge discovery and data mining', 'id': '1130985203', 'acronym': 'KDD', 'area': 'DM'},
            {'raw': 'european conference on principles of data mining and knowledge discovery', 'id': '1141769385', 'acronym': 'ECML-PKDD', 'area': 'DM'},
            {'raw': 'web search and data mining', 'id': '1120384002', 'acronym': 'WSDM', 'area': 'DM'},
            {'raw': 'pacific-asia conference on knowledge discovery and data mining', 'id': '1178327129', 'acronym': 'PAKDD', 'area': 'DM'},
            {'raw': 'international conference on data mining', 'id': '1183478919', 'acronym': 'ICDM', 'area': 'DM'},
            {'raw': 'siam international conference on data mining', 'id': '1142743330', 'acronym': 'SDM', 'area': 'DM'}]
    else:
        raise ValueError("Wrong data name.")

def load_paper_records(data_file, target_venues):
    target_vids = [venue['id'] for venue in target_venues]
    paper_records = defaultdict(dict)
    with open(data_file) as f:
        for i, line in tqdm(enumerate(f)):
            prec = json.loads(line)
            try:
                if prec['venue']['id'] in target_vids:
                    paper_records[prec['id']] = prec
            except:
                pass
    return paper_records

def load_fos_records(paper_records):
    internal_id = 0
    fos_records = dict()
    for pid, prec in paper_records.items():
        if 'fos' in prec:
            for frec in prec['fos']:
                topic = frec['name']
                if topic not in fos_records:
                    fos_records[topic] = {'id': internal_id, 'name': topic}
                    internal_id += 1
    return fos_records

def subsample_paper_records(paper_records, subsample_step):
    result = dict()
    count = 0
    for pid, prec in paper_records.items():
        if count % subsample_step == 0:
            result[pid] = prec
        count += 1
    return result

def load_dblp_data(data_name, data_file, cache_file, subsample_step):
    target_venues = load_venues(data_name)
    paper_records = memoize(load_paper_records, cache_file)(data_file,
                                                            target_venues)
    paper_records = subsample_paper_records(paper_records, subsample_step)

    fos_records = load_fos_records(paper_records)

    pid_id_dict = {pid: i for i, pid in enumerate(paper_records.keys())}
    dblp_data = {
        'data_name': data_name,
        'paper_records': paper_records,
        'fos_records': fos_records,
        'target_venues': target_venues,
        'pid_id_dict': pid_id_dict
    }
    return dblp_data

def load_dblp_data_affs(data_name, data_file, cache_file, subsample_step):
    target_venues = load_venues(data_name)
    paper_records = memoize(load_paper_records, cache_file)(data_file,
                                                            target_venues)
    paper_records = subsample_paper_records(paper_records, subsample_step)
    paper_records, attr = get_attr_with_country(paper_records)

    fos_records = load_fos_records(paper_records)

    pid_id_dict = {pid: i for i, pid in enumerate(paper_records.keys())}
    dblp_data = {
        'data_name': data_name,
        'paper_records': paper_records,
        'fos_records': fos_records,
        'target_venues': target_venues,
        'pid_id_dict': pid_id_dict
    }
    return dblp_data, attr

def compute_adj(dblp_data):
    paper_records = dblp_data['paper_records']
    pid_id_dict = dblp_data['pid_id_dict']
    E = []
    for pid, prec in paper_records.items():
        if 'references' in prec and 'fos' in prec:
            for ref_pid in prec['references']:
                if ref_pid in pid_id_dict:
                    E.append([pid_id_dict[pid], pid_id_dict[ref_pid]])
    E = np.array(E)
    n = len(dblp_data['paper_records'])

    A = sparse.csr_matrix((np.ones(len(E)), (E[:,0], E[:,1])), shape=(n,n))
    return A

def compute_attributes(dblp_data):
    paper_records = dblp_data['paper_records']
    pid_id_dict = dblp_data['pid_id_dict']
    fos_records = dblp_data['fos_records']
    E = []
    weights = []
    for pid, prec in paper_records.items():
        if 'references' in prec and 'fos' in prec:
            for frec in prec['fos']:
                topic = frec['name']
                E.append([pid_id_dict[pid], fos_records[topic]['id']])
                weights.append(frec['w'])
    E = np.array(E)
    A = sparse.csr_matrix((weights, (E[:,0], E[:,1])),
                          shape=(len(paper_records),len(fos_records)))
    return A

def compute_LSA(term_doc_matrix, n_components):
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=0)
    svd.fit(term_doc_matrix)
    return svd.components_.T

def compute_DBSCAN(LSA, eps):
    LSA_norm = (LSA.T/np.sum(LSA**2+1e-10, axis=1)**.5).T
    db = DBSCAN(eps=eps, min_samples=10, n_jobs=8).fit(LSA_norm)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    silhouette_coff = metrics.silhouette_score(LSA_norm, labels)
    print(eps, n_clusters_, n_noise_, silhouette_coff)
    return labels

def compute_KMEANS(LSA, k):
    LSA_norm = (LSA.T/np.sum(LSA**2+1e-10, axis=1)**.5).T
    kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=8).fit(LSA_norm)
    labels = kmeans.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    silhouette_coff = metrics.silhouette_score(LSA_norm, labels)
    print(k, n_clusters_, n_noise_, silhouette_coff)
    return kmeans.labels_

def get_countries_and_states():
    country_state_dict = {} # a dict map each country name to a most common name
    country_list = []
    state_abbv_list = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE',
    'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
    'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NA', 'NC', 'ND', 'NE', 'NH', 'NJ',
    'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX',
    'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']

    for state in state_abbv_list:
        country_state_dict[state]=state

    for country in list(pycountry.countries):
        country_list.append(country.name)
        country_state_dict[country.name] = country.name
        if hasattr(country,'official_name'):
            country_list.append(country.official_name)
            country_state_dict[country.official_name] = country.name

        if hasattr(country,'common_name'):
            country_list.append(country.common_name)
            country_state_dict[country.common_name] = country.common_name
            country_state_dict[country.official_name] = country.common_name
            country_state_dict[country.name] = country.common_name

    country_list.extend(['US', 'USA', 'America', 'UK'])
    country_state_dict['US'] = 'United States'
    country_state_dict['USA'] = 'United States'
    country_state_dict['America'] = 'United States'
    country_state_dict['UK'] = 'United Kingdom'
    country_state_dict['Hong Kong']= 'China'
    country_state_dict['Macao']= 'China'
    country_state_dict['Taiwan']= 'China'

    country_state_list = country_list + state_abbv_list

    return country_state_list, country_state_dict


def get_attr_with_country(paper_records):
    country_state_list, country_state_dict = get_countries_and_states()
    filtered_precs = {}
    pid_attr_dict = defaultdict(set)

    state_abbv_list = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE',
    'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
    'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NA', 'NC', 'ND', 'NE', 'NH', 'NJ',
    'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX',
    'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']

    # counter = 0
    for pid, prec in paper_records.items():
        add = False
        authors = prec['authors']
        for author in authors:
            if 'org' in author:
                # print(author['org'])
                tokenized_org_list = [e.strip(' ,')for e in author['org'].replace('#TAB#','').split()]
                # print(tokenized_org_list)
                countries = list(set(tokenized_org_list)&set(country_state_list))
                if countries !=[]:
                    add = True
                    for cntry in countries:
                        pid_attr_dict[pid].add(country_state_dict[cntry])
                        if cntry in state_abbv_list:
                            pid_attr_dict[pid].add('United States')

        # counter+=1
        # if counter >10:
        #     break
        if add == True:
            filtered_precs[pid]=prec
    attrs = np.asarray(list(set(itertools.chain(*list(pid_attr_dict.values())))))
    # attrs = np.array(list(set(pid_attr_dict.values())))
    mlb = MultiLabelBinarizer(attrs)
    attr_rec = mlb.fit_transform(list(pid_attr_dict.values()))

    id_attr_dict = {id: attr_rec[id,:] for id, pid in enumerate(filtered_precs.keys())}
    attr = pd.DataFrame.from_dict(id_attr_dict, orient='index',columns=attrs)

    return filtered_precs, attr

if __name__ == '__main__':
    data_name = 'dblp_four_areas'

    data_folder = './datasets/'
    cache_folder = mkdir('./cache/')

    raw_data_file = join(data_folder, 'dblp_papers_v11.txt')
    paper_records_cache_file = join(cache_folder, 'dblp_paper_records_affs.pkl')
    dblp_data_cache_file = join(cache_folder, 'dblp_data_affs.pkl')
    adj_A_cache_file = join(cache_folder, 'dblp_adj_A_affs.pkl')
    # att_A_cache_file = join(cache_folder, 'dblp_att_A.pkl')
    # LSA_cache_file = join(cache_folder, 'dblp_lsa.pkl')
    # DBSCAN_cache_file = join(cache_folder, 'dblp_dbscan.pkl')
    # KMEANS_cache_file = join(cache_folder, 'dblp_kmeans.pkl')
    subsample_step = 8

    dblp_data, attr = memoize(load_dblp_data_affs, dblp_data_cache_file,refresh=True)(data_name,
                            raw_data_file, paper_records_cache_file,
                            subsample_step)
    adj_A = memoize(compute_adj, adj_A_cache_file, refresh=True)(dblp_data)

    # # research topics as attributes
    # att_A = memoize(compute_attributes, att_A_cache_file, refresh=False)(dblp_data)
    #
    # LSA = memoize(compute_LSA, LSA_cache_file, refresh=False)(att_A, 50)
    # # labels = memoize(compute_DBSCAN, DBSCAN_cache_file,
    # #                  refresh=True)(LSA, 0.5)
    # labels = memoize(compute_KMEANS, KMEANS_cache_file,
    #                  refresh=True)(LSA, 200)
    # print(adj_A.shape, np.sum(adj_A))
    # print(att_A.shape, np.sum(att_A))
    # print(LSA.shape)


    # countries as compute_attributes
    print(adj_A.shape, np.sum(adj_A))
    print(attr.head(10))
    print(attr.iloc)
    print(attr.columns)

    paper_records = dblp_data['paper_records']
    pid_id_dict = dblp_data['pid_id_dict']
    id_pid_dict = {pid_id_dict[pid]: pid for pid in list(pid_id_dict.keys())}
    print(paper_records[id_pid_dict[0]]['authors'])
    a = (attr.iloc[0]==1)
    for name, row in a.items():
        print(name, row)
