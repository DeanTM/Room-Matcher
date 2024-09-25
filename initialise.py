import networkx as nx
from typing import Sequence
import random


def initial_random(
    preference_graph: nx.DiGraph,
    partition_sizes: Sequence[int],
    max_used_partitions: int
) -> dict[int,list]:
    """Finds a random solution to initialise the IP.
    
    :param preference_graph: The digraph where edges show preferences to be in the
        same partition, with edge weights specified in the `_weight_key` parameter.
    :type preference_graph: nx.DiGraph
    :param partitions_sizes: A sequence of the available partition sizes.
    :type partitions_sizes: Sequence[int]
    :param max_used_partitions: Maximum number of allowed partitions. Defaults to 
        `len(partitions_sizes)`.
    :type max_used_partitions: int

    :returns: A potentially incomplete dictionary specifying partitions and initial
        candidates. It can be used to initialise the MIP.
    :rtype: dict[int,list]
    """
    V = list(preference_graph.nodes)
    random.shuffle(V)
    partition_sizes_dict = dict(enumerate(partition_sizes))
    partitions = {}
    while len(V) > 0 and len(partitions) < max_used_partitions:
        # get size of biggest partition with available space
        max_remaining_partition_size = max(
            s for i,s in partition_sizes_dict.items()
            if len(partitions.get(i,[])) < s)
        # get all partitions of this size
        biggest_partitions = [
            i for i,s in partition_sizes_dict.items()
            if len(partitions.get(i,[])) < s
            and s == max_remaining_partition_size]
        # better to fill the big ones first
        biggest_partitions_in_use = [i for i in biggest_partitions if i in partitions]
        if len(biggest_partitions_in_use) > 0:
            biggest_partitions = biggest_partitions_in_use
        partition_idx = random.choice(biggest_partitions)
        if partition_idx not in partitions:
            partitions[partition_idx] = []
        node = V.pop()
        partitions[partition_idx].append(node)
    else:
        if len(V) > 0:
            raise RuntimeError("Failed to assign all nodes to a partition")
    return partitions


def initial_estimates(
    preference_graph: nx.DiGraph,
    partition_sizes: Sequence[int],
    max_used_partitions: int
) -> dict[int,list]:
    """Finds a feasible (suboptimal) solution to initialise the IP.
    
    :param preference_graph: The digraph where edges show preferences to be in the
        same partition, with edge weights specified in the `_weight_key` parameter.
    :type preference_graph: nx.DiGraph
    :param partitions_sizes: A sequence of the available partition sizes.
    :type partitions_sizes: Sequence[int]
    :param max_used_partitions: Maximum number of allowed partitions. Defaults to 
        `len(partitions_sizes)`.
    :type max_used_partitions: int

    :returns: A potentially incomplete dictionary specifying partitions and initial
        candidates. It can be used to initialise the MIP.
    :rtype: dict[int,list]
    """
    remaining_partition_sizes_dict = dict(enumerate(partition_sizes))
    partitions = {}
    components_left = list(nx.connected_components(preference_graph.to_undirected()))
    nodes_assigned = []
    while (len(components_left) > 0)\
        and (len(remaining_partition_sizes_dict) > 0)\
        and len(partitions) <= max_used_partitions:
        components_left = sorted(components_left, key=len, reverse=True)  # handle largest components first
        comp_vertices = components_left.pop()
        comp_subgraph = preference_graph.subgraph(comp_vertices)
        if comp_subgraph.order() <= max(remaining_partition_sizes_dict.values()):
            # partition_idx = min([x for x in remaining_partition_sizes if x >= comp_subgraph.order()])
            partition_idx = min([
                i for i in remaining_partition_sizes_dict
                if remaining_partition_sizes_dict[i] >= comp_subgraph.order()])
            del remaining_partition_sizes_dict[partition_idx]
            partitions[partition_idx] = list(comp_subgraph.nodes)
            nodes_assigned += list(comp_subgraph.nodes)
        else:
            # iteratively split with the Fiedler vector
            part1, part2 = [], []
            fiedler = nx.fiedler_vector(comp_subgraph.to_undirected())
            for v, coeff in zip(comp_subgraph.nodes, fiedler):
                if coeff >= 0:
                    part1.append(v)
                else:
                    part2.append(v)
            components_left.append( part1 )
            components_left.append( part2 )
    # remaining nodes will be assigned where they fit
    remaining_nodes = [v for v in preference_graph.nodes if v not in nodes_assigned]
    while len(remaining_nodes) > 0:
        node = remaining_nodes.pop()
        for i in partitions:
            if len(partitions[i]) < partition_sizes[i]:
                partitions[i].append(node)
                break
        else:
            raise RuntimeError("Failed to assign all nodes to a partition")
    return partitions