import re
import csv
import networkx as nx
from mip_solve import solve_mip
from ea_solve import solve_ea
from initialise import initial_estimates
from pprint import pprint

def extract_emails(text: str):
    # Define the regex pattern for matching email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Use re.findall to find all matches of the pattern in the text
    emails = re.findall(email_pattern, text)
    
    return emails

def extract_vertices_and_edges(csvfile, source_col: int=0, target_col: int=3):
    V, E = [], []
    reader = csv.reader(csvfile, delimiter="\t")
    for row in reader:
        source = extract_emails(row[source_col].lower())
        targets = extract_emails(row[target_col].lower())
        if source:
            s = source[0]
            V.append(s)
            for t in targets:
                E.append((s, t))
    return V, E


if __name__ == "__main__":
    import sys

    preferences_file = sys.argv[1]
    # Build graph
    with open(preferences_file, "r") as csvfile:
        V, E = extract_vertices_and_edges(csvfile)
    # remove out-of-graph preferences
    E = [(u,v) for (u,v) in E if v in V and u in V]
    preference_graph = nx.DiGraph()
    preference_graph.add_nodes_from(V)
    preference_graph.add_edges_from(E)

    # Check for validity
    if len(V) != preference_graph.order():
        for v in preference_graph.nodes:
            if v not in V:
                raise ValueError(f"Graph has node {v} not from vertex list.")
        for v in V:
            if v not in preference_graph.nodes:
                raise ValueError(f"Vertex {v} not not in graph.")
    assert len(V) == preference_graph.order(),\
        "Graph has different vertices to constructed vertex list."
    assert len(E) == preference_graph.size(),\
        "Graph has different edges to constructed edge list."

    # Set parameters
    partition_sizes = 44*[2] + 12*[3]
    max_used_partitions = 44
    occupancy_costs = 44*[0.2] + 12*[0.25]
    default_weights = lambda *args: 1.0
    verbose = True
    timeLimit = 10

    # Get initial estimate
    initial_solution = initial_estimates(
        preference_graph=preference_graph,
        partition_sizes=partition_sizes,
        max_used_partitions=max_used_partitions)

    # Improve estimate with PuLP
    solution, cost = solve_mip(
        preference_graph=preference_graph,
        partition_sizes=partition_sizes,
        max_used_partitions=max_used_partitions,
        initial_solution=initial_solution,
        _occupancy_costs=occupancy_costs,
        _default_weights=default_weights,
        verbose=verbose,
        timeLimit=timeLimit)
    solution_nonempty = {k:v for k,v in solution.items() if len(v) > 0}
    print(f"{cost=}, {len(solution_nonempty)=}")
    pprint(solution_nonempty)
    
    # Try improve with EA
    solution_ea, cost_ea = solve_ea(
        preference_graph=preference_graph,
        population_size=1000,
        n_generations=1000,
        move_rate=1.,
        partition_sizes=partition_sizes,
        max_used_partitions=max_used_partitions,
        initial_solution=solution,
        _occupancy_costs=occupancy_costs,
        _default_weights=default_weights,
        verbose=verbose,
        timeLimit=timeLimit,
    )
    solution_ea_nonempty = {k:v for k,v in solution.items() if len(v) > 0}
    print(f"{cost_ea=}, {len(solution_ea_nonempty)=}")
    pprint(solution_ea_nonempty)
    # sanity check that everyone is accounted for once
    print(preference_graph.order())
    print(sum(len(v) for v in solution.values()))
    print(len(set(sum(solution.values(), []))))
    print(sum(len(v) for v in solution_ea.values()))
    print(len(set(sum(solution_ea.values(), []))))
