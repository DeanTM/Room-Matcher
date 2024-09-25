import pulp as plp
import networkx as nx
from typing import Sequence, Callable


def solve_mip(
    preference_graph: nx.DiGraph,
    partition_sizes: Sequence[int],  # all rooms and their sizes, for example 40*[2] + 10*[3] for 40 2-bedrooms and 10 3-bedrooms
    max_used_partitions: int|None=None,  # if None, assume all partitions can be used
    initial_solution: dict[int,str]|None=None,
    verbose: bool=True,
    _occupancy_costs: Sequence[float]|float=0,
    _default_weights: Callable[[nx.DiGraph, tuple], float] = lambda g,e: 1.,
    _weight_key: str='weight',
    **solver_kwargs
) -> tuple[dict[str,dict], tuple[float,float]]:
    """Solves the graph partition problem on the given `preference_graph` using a
    mixed integer program.

    :param preference_graph: The digraph where edges show preferences to be in the
        same partition, with edge weights specified in the `_weight_key` parameter.
    :type preference_graph: nx.DiGraph
    :param partitions_sizes: A sequence of the available partition sizes.
    :type partitions_sizes: Sequence[int]
    :param max_used_partitions: Maximum number of allowed partitions. Defaults to 
        `len(partitions_sizes)`.
    :type max_used_partitions: int
    :param initial_solution: A candidate initial solution to seed the process.
    :type initial_solution: dict[int,str], optional
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
    :param **solver_kwargs: kwargs passed on to the default PuLP solver. For example,
        one may set `timeLimit` to force the search to stop after a specified number
        of seconds.


    ## Usage

    Assume we have 20 available rooms, 10 of which can be converted in 3-bed rooms,
    while all of them can be at-least-2-bed rooms. We let the algorithm decide whether
    to use 3 beds or two, but we provide all options. We can also assume that -bed rooms
    are slightly more expensive, but not so much that it's worth breaking preferences
    over.

    ```python
        solution, cost = solve_mip(
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
    n_partitions = len(partition_sizes)
    if max_used_partitions is None:
        max_used_partitions = n_partitions
    if isinstance(_occupancy_costs, (float,int)):
        _occupancy_costs = n_partitions*[_occupancy_costs]

    # Set up the IP Variables
    lp_problem = plp.LpProblem("PreferenceMatching", plp.LpMinimize)
    
    # Assignments: x[v,i]=1 if node v is in partition i
    x_assignment = plp.LpVariable.dicts(
        "x", [(v,i) for v in V for i in range(n_partitions)], cat='Binary')
    
    # assign initial partitions
    if initial_solution is not None:
        for i, v_list in initial_solution.items():
            for v in v_list:
                for j in range(n_partitions):
                    # Set initial assignment
                    x_assignment[(v,i)].setInitialValue( float(i==j) )
    
    # Edge cuts: e[u,v]=1 if edge (u,v) is cut between partitions
    e_edge_cut = plp.LpVariable.dicts(
        "e", E, cat='Binary')
    # e_edge_cut_by_partion = plp.LpVariable.dicts(
    #     "ei", [(e,i) for e in E for i in range(n_partitions)], cat='Binary')

    # Measure total number of used partitions
    y_partition_used = plp.LpVariable.dicts(
        "y", range(n_partitions), cat='Binary')
    
    # Set up constraints
    # Minimize the total weight of edges cut
    objective = plp.lpSum([weights[edge] * e_edge_cut[edge] for edge in E])

    # Each vertex belongs to exactly one partition
    for v in V:
        lp_problem += plp.lpSum([x_assignment[v,i] for i in range(n_partitions)]) == 1

    # Maximal partition size constraints
    for i in range(n_partitions):
        lp_problem += plp.lpSum([x_assignment[v,i] for v in V]) <= partition_sizes[i]
    
    # Edge cut constraints: e_uv = 1 if u and v are in different partitions
    for (u,v) in E:
        for i in range(n_partitions):
            # x[u,i] - x[v,i] is 1, 0, or -1
            # and 0 for all i only if x[u,i] == x[v,i] for all i.
            # Therefore (e[u,i] == 0) => (x[u,i] == x[v,i] for all i)
            lp_problem += e_edge_cut[(u,v)] >= (x_assignment[u,i] - x_assignment[v,i])
        
    # Maximum number of used partitions
    # First, ensure that y measures whether the partition is occupied
    for i in range(n_partitions):
        # The first line ensures that y[i] is 1 if at least one node is assigned to partition i
        lp_problem += plp.lpSum([x_assignment[v,i] for v in V])  <= len(V) * y_partition_used[i]
        # The second line ensures that y[i] is 0 if no node is assigned to partition i
        # This line seems redundant if the _occupancy_costs are all positive,
        # and since we're only upper-bounding the number of used partitions, we
        # only need an upper bound estimate of it if the _occupancy_costs are zero,
        # so perhaps this line can be omitted.
        lp_problem += plp.lpSum([x_assignment[v,i] for v in V])  >= y_partition_used[i]
    # Minimise the occupancy cost of the rooms, if it's non-zero
    partition_costs = plp.lpSum([y_partition_used[i] * _occupancy_costs[i] for i in range(n_partitions)])

    # Ensure that the number of used partitions is below threshold
    total_partitions_used = plp.lpSum([y_partition_used[i] for i in range(n_partitions)])
    lp_problem += total_partitions_used <= max_used_partitions

    # Assign the objective to the function
    lp_problem += objective + partition_costs, "Minimising edge cuts and partition costs"

    # Solve the problem
    # Access default solver, specify helping values:
    solver_type = type(plp.LpSolverDefault)
    kwargs = {'msg': verbose}
    kwargs.update(solver_kwargs)
    lp_problem.solve( solver_type(**kwargs) )
    
    print(lp_problem.status)
    
    # Extract results
    solution = {i: [] for i in range(n_partitions)}
    for v in V:
        for i in range(n_partitions):
            if plp.value(x_assignment[v,i]) == 1:
                solution[i].append(v)
    assert plp.value(total_partitions_used) == len([
        i for i in solution if len(solution[i]) > 0])
    return solution, (plp.value(objective), plp.value(partition_costs))


if __name__ == "__main__":
    preference_graph = nx.karate_club_graph()

    solution, cost = solve_mip(
        preference_graph,
        partition_sizes = 20*[2] + 10*[3],
        max_used_partitions = 20,
        _occupancy_costs = 20*[0.005] + 10*[0.006],
        verbose=False
    )
    print("Assignments:", {k: v for k,v in solution.items() if len(v) > 0})
    print("Num partitions used:", len([v for v in solution.values() if len(v) > 0]))
    print("Total edge cut cost:", cost)