import pulp as plp
import networkx as nx
from typing import Sequence, Callable

def solve_ip(
    preference_graph: nx.DiGraph,
    partition_sizes: Sequence[int],  # all rooms and their sizes, for example 40*[2] + 10*[3] for 40 2-bedrooms and 10 3-bedrooms
    max_used_partitions: int|None=None,  # if None, assume all partitions can be used
    verbose: bool=True,
    _occupancy_costs: Sequence[float]|float=0,
    _default_weights: Callable[[nx.DiGraph, tuple], float] = lambda g,e: 1.,
    _weight_key: str='weight',
) -> dict[str,dict]:
    """Solves the graph partition problem on the given `preference_graph`.

    :param preference_graph: The digraph where edges show preferences to be in the
        same partition, with edge weights specified in the `_weight_key` parameter.
    :type preference_graph: nx.DiGraph
    :param partitions_sizes: A sequence of the available partition sizes.
    :type partitions_sizes: Sequence[int]
    :param max_used_partitions: Maximum number of allowed partitions. Defaults to 
        `len(partitions_sizes)`.
    :type max_used_partitions: int
    :param verbose: Whether to print solver steps. Defaults to True.
    :type verbose: bool
    :param _occupancy_costs: The cost of each occupied room (relative to the preference
        weights). Defaults to 0.
    :type _occupancy_costs: Sequence[float] | float
    :param _default_weights: A function that accepts the graph and an edge, and returns
        the edge weight, in the instance that the weights at `_weight_key` are not
        available. Defaults to a function returning 1.
    :type _default_weights: Callable[[DiGraph, tuple], float],
    :param _weight_key: The key for the weight in the graph's edge data. Defaults to
        "weight".
    :type _weight_key: str


    ## Usage

    Assume we have 20 available rooms, 10 of which can be converted in 3-bed rooms,
    while all of them can be at-least-2-bed rooms. We let the algorithm decide whether
    to use 3 beds or two, but we provide all options. We can also assume that -bed rooms
    are slightly more expensive, but not so much that it's worth breaking preferences
    over.

    ```python
        solution, cost = solve_ip(
            preference_graph,
            partition_sizes = 20*[2] + 10*[3],
            max_used_partitions = 20,
            _occupancy_costs = 20*[0.002] + 10*[0.003]
        )
    ```
    """
    V = list(preference_graph.nodes)
    E = list(preference_graph.edges)
    weights = {
        edge: preference_graph[edge[0]][edge[1]].get(
            _weight_key,
            _default_weights(preference_graph, edge))
        for edge in E}
    # print(weights)
    # n_partitions = sum(max_partition_sizes.values())
    n_partitions = len(partition_sizes)
    if max_used_partitions is None:
        max_used_partitions = n_partitions
    if isinstance(_occupancy_costs, (float,int)):
        _occupancy_costs = n_partitions*[_occupancy_costs]

    # Set up the IP Variables
    prob = plp.LpProblem("Preference_Matching", plp.LpMinimize)
    # Assignments: x[v,i]=1 if node v is in partition i
    x_assignment = plp.LpVariable.dicts(
        "x", [(v,i) for v in V for i in range(n_partitions)], cat='Binary')
    # Edge cuts: e[u,v]=1 if edge (u,v) is cut between partitions
    e_edge_cut = plp.LpVariable.dicts(
        "e", E, cat='Binary')
    # Measure total number of used partitions
    y_partition_used = plp.LpVariable.dicts(
        "y", range(n_partitions), cat='Binary')
    
    # Set up constraints
    # Minimize the total weight of edges cut
    objective = plp.lpSum([weights[edge] * e_edge_cut[edge] for edge in E])
    prob += objective
    # Each vertex belongs to exactly one partition
    for v in V:
        prob += plp.lpSum([x_assignment[v,i] for i in range(n_partitions)]) == 1
    # Maximal partition size constraints
    for i in range(n_partitions):
        prob += plp.lpSum([x_assignment[v,i] for v in V]) <= partition_sizes[i]
    # Edge cut constraints: e_uv = 1 if u and v are in different partitions
    for (u, v) in E:
        prob += e_edge_cut[(u, v)] >= plp.lpSum(
            [x_assignment[u, i] - x_assignment[v, i] for i in range(n_partitions)])
    # Maximum number of used partitions
    # First, ensure that y measures whether the partition is occupied
    for i in range(n_partitions):
        # The first line ensures that y[i] is 1 if at least one node is assigned to partition i
        prob += plp.lpSum([x_assignment[v,i] for v in V])  <= len(V) * y_partition_used[i]
    # The second line ensures that y[i] is minimised i.e. that it is 0 if no node is assigned to partition i
    # which also indirectly minimises the number of used partitions, so we assign an occupancy cost
    # If the occupancy cost is zero, the number of used rooms is only bounded by `max_used_partitions`,
    # and so may be overestimated. Should should have no effect on the final assignment, however.
    prob += plp.lpSum([y_partition_used[i] * _occupancy_costs[i] for i in range(n_partitions)])
    total_partitions_used = plp.lpSum([y_partition_used[i] for i in range(n_partitions)])
    # Ensure that the number of used partitions is below our threshold
    prob += total_partitions_used <= max_used_partitions

    # Solve the problem
    if not verbose:
        # Get the default solver, and create a new instance with `msg` as False
        prob.solve( type(plp.LpSolverDefault)(msg=False) )
    else:
        prob.solve()
    
    
    # Extract results
    partitions = {i: [] for i in range(n_partitions)}
    for v in V:
        for i in range(n_partitions):
            if plp.value(x_assignment[v,i]) == 1:
                partitions[i].append(v)
    return partitions, plp.value(objective)


if __name__ == "__main__":
    preference_graph = nx.karate_club_graph()

    solution, cost = solve_ip(
        preference_graph,
        partition_sizes = 20*[2] + 10*[3],
        max_used_partitions = 20,
        _occupancy_costs = 20*[0.005] + 10*[0.006],
        verbose=False
    )
    print("Assignments:", {k: v for k,v in solution.items() if len(v) > 0})
    print("Num partitions used:", len([v for v in solution.values() if len(v) > 0]))
    print("Total edge cut cost:", cost)