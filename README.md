# GlobalESSD_Public

Source code for paper: Mining Explainable Subgraphs with Surprising Densities, Locally and Globally

## Set up

*  Environment: Python 3.6+
*  Requirement:
    *  numpy 1.15.2
    *  scipy 1.1.0
    *  pandas 0.23.4
    *  matplotlib 3.0.0
    *  networkx 2.2
    *  scikit-learn 0.20.0
    *  graphviz 0.10.1

## Data

We run experiments on several real-world networks, including:

*  **lastfm**:
   from the website http://www.lastfm.com
   [1]Iván Cantador, Peter Brusilovsky, and Tsvi Ku ik. 2011. 2nd Workshop on  Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011).  In Proceedings of the 5th ACM conference on Recommender systems (RecSys2011).

*  **facebook100**:
   Caltech36, Reed98;
   [2]Amanda L. Traud, Peter J. Mucha, Mason A. Porter, Social structure of Facebook networks, Physica A: Statistical Mechanics and its Applications,
   Volume 391, Issue 16, 2012,Pages 4165-4180, ISSN 0378-4371, https://doi.org/10.1016/j.physa.2011.12.021. (http://www.sciencedirect.com/science/article/pii/S0378437111009186)

*  **DBLP**:
   DBLPaffs, DBLPtopics;
   two DBLP citation networks extracted from https://aminer.org/citation
   dblp_papers_v11.txt

*  **MPvotes**:
   the Twitter social network generated from friendships between Members of Parliament (MPs) in UK.


## Run


*  **Single-subgroup pattern discovery using our SI measure**:  
    run script 'single_sgd.py' to generate results


*  **Single-subgroup pattern discovery using other objective measures for a comparative study**:
    run script 'single_sgd_compare.py' to identify most interesting single-subgroup patterns w.r.t other different measures


*  **Bi-subgroup pattern discovery using our SI measures**:
    run script 'bi_sgd.py' to generate results


*  **Bi-subgroup pattern discovery using our SI measures, based on incorporating the user’s prior knowledge about ‘year’, ‘dorm/house’  
    attributes in either Caltech36 or Reed98 dataset**:
    run script 'bi_sgd_attrPrior.py'


*  **Global pattern discovery (summarization) on DblpAffs**:
   run script 'find_citation_patterns_globally_affs.py'


*  **Global pattern discovery (summarization) on DblpTopics**:
   run script 'find_citation_patterns_globally_topics.py'


*  **Global pattern discovery (summarization) on MPvotes**:
   run script 'find_patterns_globally_MP.py'

*  **Scalability evaluation: the effect of |S|**:
    *  single-subgroup pattern discovery: run script 'scalability_single_sgd.py'
    *  bi-subgroup pattern discovery: run script 'scalability_bi_sgd.py'
    *  global pattern discovery (summarization): run script 'scalability_global_sgd.py'

## Do not contribute to

* maxent


## Partially contribute to

* pysubgroupx (we create graph_target.py, graph_target_blockcons.py, insert the nested beam search on the algorithm.py and modify utils.py, )
