
import copy
from heapq import heappush, heappop
from itertools import islice
from typing import List

import pysubgroupx.numeric_target as nt
import pysubgroupx.measures as m
import pysubgroupx.utils as ut
from pysubgroupx.subgroup import Subgroup, SubgroupDescription
from pysubgroupx.subgroup import createNominalSelectorsForAttribute
from pysubgroupx.subgroup import createNumericSelectorForAttribute


import numpy as np
import functools


class SubgroupDiscoveryTask(object):
    '''
    Capsulates all parameters required to perform standard subgroup discovery
    '''
    def __init__(self, data, target, searchSpace, qf, resultSetSize=[10], depth=3, minQuality=0, weightingAttribute=None):
            self.data = data
            self.target = target
            self.searchSpace = searchSpace
            self.qf = qf
            self.resultSetSize = resultSetSize
            self.depth = depth
            self.minQuality = minQuality
            self.weightingAttribute = weightingAttribute


class Apriori(object):
    def execute(self, task):
        measure_statistics_based = hasattr(task.qf, 'optimisticEstimateFromStatistics')
        result = []

        # init the first level
        next_level_candidates = []
        for sel in task.searchSpace:
            next_level_candidates.append (Subgroup(task.target, [sel]))

        # level-wise search
        depth = 1
        while (next_level_candidates):
            # check sgs from the last level
            promising_candidates = []
            for sg in next_level_candidates:
                if (measure_statistics_based):
                    statistics = sg.get_base_statistics(task.data)
                    ut.addIfRequired (result, sg, task.qf.evaluateFromStatistics (*statistics), task)
                    optimistic_estimate = task.qf.optimisticEstimateFromStatistics (*statistics) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
                else:
                    ut.addIfRequired (result, sg, task.qf.evaluateFromDataset(task.data, sg), task)
                    optimistic_estimate = task.qf.optimisticEstimateFromDataset(task.data, sg) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")

                # optimistic_estimate = task.qf.optimisticEstimateFromDataset(task.data, sg) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
                # quality = task.qf.evaluateFromDataset(task.data, sg)
                # ut.addIfRequired (result, sg, quality, task)
                if (optimistic_estimate >= ut.minimumRequiredQuality(result, task)):
                    promising_candidates.append(sg.subgroupDescription.selectors)

            if (depth == task.depth):
                break

            # generate candidates next level
            next_level_candidates = []
            for i, sg1 in enumerate(promising_candidates):
                for j, sg2 in enumerate (promising_candidates):
                    if (i < j and sg1 [:-1] == sg2 [:-1]):
                        candidate = list(sg1) + [sg2[-1]]
                        # check if ALL generalizations are contained in promising_candidates
                        generalization_descriptions = [[x for x in candidate if x != sel] for sel in candidate]
                        if all (g in promising_candidates for g in generalization_descriptions):
                            next_level_candidates.append(Subgroup (task.target, candidate))
            depth = depth + 1

        result.sort(key=lambda x: x[0], reverse=True)
        return result

class BestFirstSearch (object):
    def execute(self, task):
        result = []
        queue = []
        measure_statistics_based = hasattr(task.qf, 'optimisticEstimateFromStatistics')


        # init the first level
        for sel in task.searchSpace:
            queue.append ((float("-inf"), [sel]))

        while (queue):
            q, candidate_description = heappop(queue)
            q = -q
            if (q) < ut.minimumRequiredQuality(result, task):
                break

            sg = Subgroup (task.target, candidate_description)

            if (measure_statistics_based):
                statistics = sg.get_base_statistics(task.data)
                ut.addIfRequired (result, sg, task.qf.evaluateFromStatistics (*statistics), task)
                optimistic_estimate = task.qf.optimisticEstimateFromStatistics (*statistics) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
            else:
                ut.addIfRequired (result, sg, task.qf.evaluateFromDataset(task.data, sg), task)
                optimistic_estimate = task.qf.optimisticEstimateFromDataset(task.data, sg) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")

            # compute refinements and fill the queue
            if (len (candidate_description) < task.depth and optimistic_estimate >= ut.minimumRequiredQuality(result, task)):
                # iterate over all selectors that are behind the last selector contained in the evaluated candidate according to the initial order
                index_of_last_selector = min (task.searchSpace.index(candidate_description[-1]), len (task.searchSpace) - 1)

                for sel in islice(task.searchSpace, index_of_last_selector + 1, None):
                    new_description = candidate_description + [sel]
                    heappush(queue, (-optimistic_estimate, new_description))
        result.sort(key=lambda x: x[0], reverse=True)
        return result

class BeamSearch(object):
    '''
    Implements the BeamSearch algorithm. Its a basic implementation without any optimization, i.e., refinements get tested multiple times.
    '''
    def __init__(self, beamWidth=30, beamWidthAdaptive=False):
        self.beamWidth = beamWidth
        self.beamWidthAdaptive = beamWidthAdaptive

    def execute (self, task):
        # adapt beam width to the result set size if desired
        if self.beamWidthAdaptive:
            self.beamWidth = task.resultSetSize[0]

        # check if beam size is to small for result set
        if (self.beamWidth < task.resultSetSize[0]):
            raise RuntimeError('Beam width in the beam search algorithm is smaller than the result set size!')

        # init
        beam = [(0, Subgroup(task.target, []))]
        last_beam = None

        depth = 0
        while beam != last_beam and depth < task.depth:
            last_beam = beam.copy()
            for (_, last_sg) in last_beam:
                for sel in task.searchSpace:
                    # create a clone
                    new_selectors = list(last_sg.subgroupDescription.selectors)

                    if not sel in new_selectors:
                        new_selectors.append(sel)

                        sg = Subgroup(task.target, new_selectors)

                        quality = task.qf.evaluateFromDataset(task.data, sg)

                        ut.addIfRequired(beam, sg, quality, task, check_for_duplicates=True)
            depth += 1


        result = beam[:task.resultSetSize[0]]
        result.sort(key=lambda x: x[0], reverse=True)
        return result


    def execute_in_bicases (self, task):
        # adapt beam width to the result set size if desired
        if self.beamWidthAdaptive:
            self.beamWidth = task.resultSetSize[0]

        # check if beam size is to small for result set
        if (self.beamWidth < task.resultSetSize[0]):
            raise RuntimeError('Beam width in the beam search algorithm is smaller than the result set size!')

        # init
        beam = [(0, Subgroup(task.target, []), Subgroup(task.target, []))]
        beam2 = [(0, Subgroup(task.target, []))]

        last_beam = None

        depth1 = 0
        depth2 = 0

        while beam!= last_beam and depth1 < task.depth:

            last_beam = beam.copy()
            last_candidates = [pattern[1] for pattern in last_beam]
            all_sg1_candidates = last_candidates

            # a list contains all sg1 without duplicates
            all_sg1_candidates = [i for n,i in enumerate(last_candidates) if i not in last_candidates[:n]]

            # print(all_sg1_candidates)

            for last_sg1 in all_sg1_candidates:

                for sel1 in task.searchSpace:
                    new_selectors1 = list(last_sg1.subgroupDescription.selectors)

                    if not sel1 in new_selectors1:
                        new_selectors1.append(sel1)
                        sg1 = Subgroup(task.target, new_selectors1)
                        # print('sg1')
                        # print(sg1)

                        if sg1.count(task.data) != 0:

                            beam2 = [(0, Subgroup(task.target, []))]
                            depth2 = 0
                            last_beam2 = None
                            searchSpace_sg2 = [i for i in task.searchSpace if i not in new_selectors1]

                            while beam2!= last_beam2 and depth2 < task.depth:
                                last_beam2 = beam2.copy()

                                for (_, last_sg2) in last_beam2:

                                    for sel2 in searchSpace_sg2:

                                        new_selectors2 = list(last_sg2.subgroupDescription.selectors)
                                        if not sel2 in new_selectors2:

                                            new_selectors2.append(sel2)

                                            # if new_selectors2 != new_selectors1:
                                            sg2 = Subgroup(task.target, new_selectors2)

                                            quality = task.qf.evaluateFromDataset_BiCases(task.data, sg1, sg2)
                                            ut.addIfRequired(beam2, sg2, quality, task, check_for_duplicates=True)

                                depth2 += 1


                            beam2.sort(key=lambda x: x[0], reverse=True)
                            # print(beam2)

                            for i in range(len(beam2)):
                                ut.addIfRequired_Bicases(beam, sg1, beam2[i][-1], beam2[i][0], task, check_for_duplicates=True)


                            # print('here')
                            # for i in range(len(beam)):
                            #     print(beam[i][1])


            depth1 += 1

        result = beam[:]
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def execute_in_bicases_constraints (self, task):
        # adapt beam width to the result set size if desired
        if self.beamWidthAdaptive:
            self.beamWidth = task.resultSetSize[0]

        # check if beam size is to small for result set
        if (self.beamWidth < task.resultSetSize[0]):
            raise RuntimeError('Beam width in the beam search algorithm is smaller than the result set size!')

        # init
        beam = [(0, Subgroup(task.target, []), Subgroup(task.target, []))]
        beam2 = [(0, Subgroup(task.target, []))]

        last_beam = None

        depth1 = 0
        depth2 = 0

        while beam!= last_beam and depth1 < task.depth:

            last_beam = beam.copy()
            last_candidates = [pattern[1] for pattern in last_beam]
            all_sg1_candidates = last_candidates

            # a list contains all sg1 without duplicates
            all_sg1_candidates = [i for n,i in enumerate(last_candidates) if i not in last_candidates[:n]]

            # print(all_sg1_candidates)
            for last_sg1 in all_sg1_candidates:


                for sel1 in task.searchSpace:

                    new_selectors1 = list(last_sg1.subgroupDescription.selectors)

                    if not sel1 in new_selectors1:
                        new_selectors1.append(sel1)
                        sg1 = Subgroup(task.target, new_selectors1)
                        # print('sg1')
                        # print(sg1)

                        if sg1.count(task.data) != 0:

                            beam2 = [(0, Subgroup(task.target, []))]
                            depth2 = 0
                            last_beam2 = None

                            searchSpace_sg2_1 = []
                            ignore_attrs = []

                            for sel in new_selectors1:
                                ignore_attr_name = sel.getAttributeName()
                                searchSpace_sg2_1.extend(createNominalSelectorsForAttribute(task.data, \
                                ignore_attr_name))
                                # searchSpace_sg2_1.extend(createNumericSelectorForAttribute(task.data, \
                                # ignore_attr_name))

                                ignore_attrs.append(ignore_attr_name)

                                searchSpace_sg2_1.remove(sel)


                            searchSpace_sg2_left = [i for i in task.searchSpace if i.getAttributeName() not in ignore_attrs]

                            map_searchSpace_sg2 = {}
                            map_searchSpace_sg2[0] = searchSpace_sg2_1


                            for i in range(1,task.depth):
                                map_searchSpace_sg2[i] = searchSpace_sg2_left

                            # print(map_searchSpace_sg2)

                            while beam2!= last_beam2 and depth2 < task.depth:
                                last_beam2 = beam2.copy()

                                for (_, last_sg2) in last_beam2:

                                    if depth2 == 0 or list(last_sg2.subgroupDescription.selectors)!=[]:
                                        # print(last_sg2)

                                        for sel2 in map_searchSpace_sg2[depth2]:

                                            new_selectors2 = list(last_sg2.subgroupDescription.selectors)
                                            new_selectors2.append(sel2)

                                            # if new_selectors2 != new_selectors1:
                                            sg2 = Subgroup(task.target, new_selectors2)
                                            quality = task.qf.evaluateFromDataset_BiCases(task.data, sg1, sg2)
                                            # print(quality)
                                            ut.addIfRequired(beam2, sg2, quality, task, check_for_duplicates=True)

                                # print(beam2)

                                depth2 += 1


                            beam2.sort(key=lambda x: x[0], reverse=True)
                            # print(beam2)

                            for i in range(len(beam2)):
                                ut.addIfRequired_Bicases(beam, sg1, beam2[i][-1], beam2[i][0], task, check_for_duplicates=True)

                            # print('here')
                            # for i in range(len(beam)):
                            #     print(beam[i][1])


            depth1 += 1

        result = beam[:]
        result.sort(key=lambda x: x[0], reverse=True)
        return result




class SimpleDFS(object):
    def execute (self, task, useOptimisticEstimates=True):
        result = self.searchInternal(task, [], task.searchSpace, [], useOptimisticEstimates)
        result.sort(key=lambda x: x[0], reverse=True)
        return result


    def searchInternal(self, task: SubgroupDiscoveryTask, prefix: List, modificationSet: List, result: List, useOptimisticEstimates: bool) -> List:
        sg = Subgroup(task.target, SubgroupDescription(copy.copy(prefix)))
        optimisticEstimate = float("inf")
        if useOptimisticEstimates and len(prefix) < task.depth and isinstance(task.qf, m.BoundedInterestingnessMeasure):
            optimisticEstimate = task.qf.optimisticEstimateFromDataset(task.data, sg)
            if (optimisticEstimate <= ut.minimumRequiredQuality(result, task)):
                return result

        if task.qf.supportsWeights():
            quality = task.qf.evaluateFromDataset(task.data, sg, task.weightingAttribute)
        else:
            quality = task.qf.evaluateFromDataset(task.data, sg)
        ut.addIfRequired (result, sg, quality, task)

        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                newModificationSet.pop(0)
                self.searchInternal(task, prefix, newModificationSet, result, useOptimisticEstimates)
                # remove the sel again
                prefix.pop(-1)
        return result


class BSD (object):
    """
    Implementation of the BSD algorithm for binary targets. See
    Lemmerich, Florian, Mathias Rohlfs, and Martin Atzmueller. "Fast Discovery of Relevant Subgroup Patterns." FLAIRS Conference. 2010.
    """
    def execute(self, task):
        self.popSize = len(task.data)
        self.targetBitset = task.target.covers(task.data)
        self.popPositives = self.targetBitset.sum()
        self.bitsets = {}
        for sel in task.searchSpace:
            self.bitsets[sel] = sel.covers(task.data).values

        result = self.searchInternal(task, [], task.searchSpace, [], np.ones(self.popSize, dtype=bool))
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def searchInternal(self, task, prefix, modificationSet, result, bitset):

        sgSize = bitset.sum()
        positiveInstances = np.logical_and(bitset, self.targetBitset)
        sgPositiveCount = positiveInstances.sum()

        optimisticEstimate = task.qf.optimisticEstimateFromStatistics(self.popSize, self.popPositives, sgSize,
                                                                      sgPositiveCount)
        if (optimisticEstimate <= ut.minimumRequiredQuality(result, task)):
            return result

        sg = Subgroup(task.target, copy.copy(prefix))

        quality = task.qf.evaluateFromStatistics(self.popSize, self.popPositives, sgSize, sgPositiveCount)
        ut.addIfRequired(result, sg, quality, task)

        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                newBitset = np.logical_and(bitset, self.bitsets[sel])
                newModificationSet.pop(0)
                self.searchInternal(task, prefix, newModificationSet, result, newBitset)
                # remove the sel again
                prefix.pop(-1)
        return result


class TID_SD (object):
    """
    Implementation of a depth-first-search with look-ahead using vertical ID lists as data structure.
    """

    def execute(self, task, use_sets=False):
        self.popSize = len(task.data)

        # generate target bitset
        x = task.target.covers(task.data)
        if use_sets:
            self.targetBitset = set (x.nonzero()[0])
        else:
            self.targetBitset = list(x.nonzero()[0])


        self.popPositives = len(self.targetBitset)

        # generate selector bitsets
        self.bitsets = {}
        for sel in task.searchSpace:
            # generate data structure
            x = task.target.covers(task.data)
            if use_sets:
                selBitset = set (x.nonzero()[0])
            else:
                selBitset = list(x.nonzero()[0])
            self.bitsets[sel] = selBitset
        if use_sets:
            result = self.searchInternal(task, [], task.searchSpace, [], set(range(self.popSize)), use_sets)
        else:
            result = self.searchInternal(task, [], task.searchSpace, [], list(range(self.popSize)), use_sets)
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def searchInternal(self, task, prefix, modificationSet, result, bitset, use_sets):

        sgSize = len(bitset)
        if use_sets:
            positiveInstances = bitset & self.targetBitset
        else:
            positiveInstances = ut.intersect_of_ordered_list(bitset, self.targetBitset)
        sgPositiveCount = len(positiveInstances)

        optimisticEstimate = task.qf.optimisticEstimateFromStatistics(self.popSize, self.popPositives, sgSize,
                                                                      sgPositiveCount)
        if (optimisticEstimate <= ut.minimumRequiredQuality(result, task)):
            return result

        sg = Subgroup(task.target, copy.copy(prefix))

        quality = task.qf.evaluateFromStatistics(self.popSize, self.popPositives, sgSize, sgPositiveCount)
        ut.addIfRequired(result, sg, quality, task)

        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                if use_sets:
                    newBitset = bitset & self.bitsets[sel]
                else:
                    newBitset = ut.intersect_of_ordered_list(bitset, self.bitsets[sel])
                newModificationSet.pop(0)
                self.searchInternal(task, prefix, newModificationSet, result, newBitset, use_sets)
                # remove the sel again
                prefix.pop(-1)
        return result

class DFS_numeric (object):
    def execute(self, task):
        if not isinstance (task.qf, nt.StandardQF_numeric):
            NotImplemented("BSD_numeric so far is only implemented for StandardQF_numeric")
        self.popSize = len(task.data)
        sorted_data =  task.data.sort_values(task.target.getAttributes(), ascending=False)

        # generate target bitset
        self.target_values = sorted_data[task.target.getAttributes()[0]].values

        f = functools.partial(task.qf.evaluateFromStatistics, len(sorted_data), self.target_values.mean())
        self.f = np.vectorize(f)

        # generate selector bitsets
        self.bitsets = {}
        for sel in task.searchSpace:
            # generate bitset
            self.bitsets[sel] = sel.covers(sorted_data).values
        result = self.searchInternal(task, [], task.searchSpace, [], np.ones(len(sorted_data), dtype=bool))
        result.sort(key=lambda x: x[0], reverse=True)

        return result

    def searchInternal(self, task, prefix, modificationSet, result, bitset):
        sgSize = bitset.sum()
        if sgSize == 0:
            return
        target_values_sg = self.target_values[bitset]

        target_values_cs = np.cumsum(target_values_sg)
        mean_values_cs = target_values_cs / (np.arange (len(target_values_cs)) +1)
        qualities = self.f (np.arange (len(target_values_cs)) + 1, mean_values_cs)
        optimisticEstimate = np.max(qualities)

        if (optimisticEstimate <= ut.minimumRequiredQuality(result, task)):
            return result

        sg = Subgroup(task.target, copy.copy(prefix))

        quality = qualities[-1]
        ut.addIfRequired(result, sg, quality, task)

        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                newBitset = bitset & self.bitsets[sel]
                newModificationSet.pop(0)
                self.searchInternal(task, prefix, newModificationSet, result, newBitset)
                # remove the sel again
                prefix.pop(-1)
        return result
