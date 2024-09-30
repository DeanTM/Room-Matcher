import copy
import random
from deap import base, creator, tools
import networkx as nx
import numpy as np
from typing import Sequence, Callable
import multiprocessing as mp
from initialise import initial_estimates, initial_random
from tqdm.auto import tqdm
import time

def compute_cut_cost(
    partitions: list|dict,
    preference_graph: nx.DiGraph,
    _default_weights: Callable[[nx.DiGraph, tuple], float] = lambda g,e: 1.,
    _weight_key: str='weight',
) -> float:
    # allow this function to be called on outputs from initialise.py
    if isinstance(partitions, dict):
        partitions = list(partitions.values())
    cost = 0.
    for partition in partitions:
        for node in partition:
            for neighbour in preference_graph[node]:
                if neighbour not in partition:
                    cost += preference_graph[node][neighbour].get(
                        _weight_key,
                        _default_weights(preference_graph, (node,neighbour)))
    return cost

def compute_occupancy_cost(
    partitions: list|dict,
    _occupancy_costs: Sequence[float]|float=0,
) -> float:
    # allow this function to be called on outputs from initialise.py
    if isinstance(partitions, list):
        partitions = dict(enumerate(partitions))
    if isinstance(_occupancy_costs, (float,int)):
        # a dictionary would be simpler, but I want consistency with `solve_mip`
        _occupancy_costs = max(partitions.keys())*[_occupancy_costs]
    cost = sum(_occupancy_costs[i] for i, p in partitions.items() if len(p) > 0)
    return cost


def _default_weights(g,e):
    """This is a function because multiprocessing cannot pickle lambda expressions."""
    return 1.

def evaluate(
    partitions: list,
    preference_graph: nx.DiGraph,
    _default_weights: Callable[[nx.DiGraph, tuple], float] = _default_weights,
    _weight_key: str='weight',
    _occupancy_costs: Sequence[float]|float=0,
) -> float:
    cut_cost = compute_cut_cost(
        partitions=partitions,
        preference_graph=preference_graph,
        _default_weights=_default_weights,
        _weight_key=_weight_key)
    occupancy_cost = compute_occupancy_cost(
        partitions=partitions,
        _occupancy_costs=_occupancy_costs)
    return cut_cost, occupancy_cost


def move_elements(
    partitions: list[list],
    partition_sizes: Sequence[int],
    max_used_partitions: int,
    num_to_move: int=1
) -> list[list]:
    # We will implicitly rely on the mutability of our partitions list
    # but we don't want to accidentally change the inputs
    partitions = copy.deepcopy(partitions)
    # remove elements from partitions
    elements_to_move = []
    for _ in range(num_to_move):
        # recompute non_empty_partitions each time because some of 
        # them may become empty after calling .pop
        non_empty_partitions = [part for part in partitions if len(part) > 0]
        selected_partition = random.choice(non_empty_partitions)
        elements_to_move.append(selected_partition.pop())
    # place elements in new random partitions without
    # violating the maximum number used
    available_partitions = {}
    # add all partially full partitions
    for i, partition in enumerate(partitions):
        if len(partition) > 0 and len(partition) < partition_sizes[i]:
            available_partitions[i] = partition
    # see how many spots remain
    non_empty_partitions = [part for part in partitions if len(part) > 0]
    remaining_spots = max_used_partitions - len(non_empty_partitions)
    # fill remaining partition slots with empty partitions
    for _ in range(remaining_spots):
        empty_partition_idx, empty_partition = random.choice(
            [(i,part) for i,part in enumerate(partitions) if len(part) == 0])
        available_partitions[empty_partition_idx] = empty_partition
    # iteratively assign elements
    for el in elements_to_move:
        try:
            partition_idx = random.choice([
                i for i,part in available_partitions.items()
                if len(part) < partition_sizes[i] ])
        except IndexError:
            raise RuntimeError(
                "No available partitions for element to move to.")
        available_partitions[partition_idx].append(el)
    return partitions


def shuffle_elements(
    partitions: list[list],
    num_to_shuffle: int=2
) -> list[list]:
    partitions = copy.deepcopy(partitions)
    # Step 1: select up to num_to_shuffle-many non-empty partitions
    selected_partitions = random.choices(
        partitions,
        k=min(len(partitions),num_to_shuffle))
    # Step 2: pop a candidate from each selected partition
    candidates = []
    subselected_partitions = []  # separate list, of the same length as `candidates`
    for partition in selected_partitions:
        # in case of multiply-selected partitions becoming empty,
        # we first ensure that `partition` is non-empty:
        if partition:
            candidate = partition.pop(random.randint(0,len(partition)-1))
            candidates.append(candidate)
            subselected_partitions.append(partition)
    # Step 3: shuffle the popped candidates
    random.shuffle(candidates)
    # Step 4: put the candidates back into the partition
    for c,p in zip(candidates, subselected_partitions):
        p.append(c)
    return partitions


def _mutate(
    partitions: list[list],
    partition_sizes: Sequence[int],
    max_used_partitions: int,
    num_to_move: int=1,
    num_to_shuffle: int=2
) -> list[list]:
    partitions = move_elements(
        partitions=partitions,
        partition_sizes=partition_sizes,
        max_used_partitions=max_used_partitions,
        num_to_move=num_to_move)
    partitions = shuffle_elements(
        partitions=partitions,
        num_to_shuffle=num_to_shuffle)
    return partitions


def solve_ea(
    preference_graph: nx.DiGraph,
    partition_sizes: Sequence[int],  # all rooms and their sizes, for example 40*[2] + 10*[3] for 40 2-bedrooms and 10 3-bedrooms
    population_size: int,
    n_generations: int,
    max_used_partitions: int|None=None,  # if None, assume all partitions can be used
    initial_solution: dict[int,str]|None=None,
    verbose: bool=True,
    timeLimit: float|None=None,
    move_rate: float=1.,
    shuffle_rate: float=2.,
    _occupancy_costs: Sequence[float]|float=0,
    _default_weights: Callable[[nx.DiGraph, tuple], float] = _default_weights,
    _weight_key: str='weight',
    _hall_of_fame_size: int=1,
    _num_workers: int=0,
    _return_hof: bool=False
) -> tuple[dict[str,dict], tuple[float,float]]|creator.HallOfFame:
    """Solves the graph partition problem on the given `preference_graph` using a 
    simple evolutionary algorithm (without crossover).

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
    :param timeLimit: If given, gives a threshold in seconds after which the evolution
        will stop.
    :type timeLimit: float, optional
    :param move_rate: The Poisson rate for moving elements between different sets in
        the partition. Defaults to 1. i.e. on average, one element is moved within each
        candidate solution at every generation.
    :type move_rate: float
    :param shuffle_rate: The Poisson rate for shuffling elements between different sets in
        the partition. Defaults to 2. i.e. on average, two elements are shuffled between 
        two partitions.
    :type shuffle_rate: float
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
    :param _hall_of_fame_size: Number of candidates to maintain in the hall of fame. 
        Defaults to 1 because the hall of fame is not returned.
    :type _hall_of_fame_size: int
    :param _num_workers: Number of multiprocessing pools to use when evaluating operations
        over the population. Because of parallel overhead, this usually seems to be slower.
        Hence it defaults to 0, but is kept for completeness.
    :type _num_workers: int


    ## Usage

    Assume we have 20 available rooms, 10 of which can be converted in 3-bed rooms,
    while all of them can be at-least-2-bed rooms. We let the algorithm decide whether
    to use 3 beds or two, but we provide all options. We can also assume that -bed rooms
    are slightly more expensive, but not so much that it's worth breaking preferences
    over.

    ```python
        solution, cost = solve_ea(
            preference_graph,
            partition_sizes = 20*[2] + 10*[3],
            max_used_partitions = 20,
            _occupancy_costs = 20*[0.002] + 10*[0.003]
        )
    ```
    """
    if timeLimit is not None:
        start_time = time.time()
    
    if _hall_of_fame_size is None:
        _hall_of_fame_size = 1
    
    # Create base classes
    # implement as multi-objective optimisation
    creator.create("FitnessMin", base.Fitness, weights=(-1, -1))
    # partition lists will be converted back to a dictionary at the end
    # for consistency with elsewhere and easier printing
    # but for now, avoid subclassing dict
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    if initial_solution is None:
        def _create_individual():
            partitions = initial_random(
                preference_graph=preference_graph,
                partition_sizes=partition_sizes,
                max_used_partitions=max_used_partitions)
            # ensure that they are sorted:
            return creator.Individual(partitions.get(i, []) for i in range(len(partition_sizes)))
    else:
        def _create_individual():
            # initial_solution = copy.deepcopy(initial_solution)
            return creator.Individual(
                initial_solution.get(i, []) for i in range(len(partition_sizes)))

    # create collection of helper operations / functions
    toolbox = base.Toolbox()
    toolbox.register("individual", _create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", _mutate,
                     max_used_partitions=max_used_partitions,
                     partition_sizes=partition_sizes)

    # def _evaluate(partitions: list) -> float:
    #     cut_cost = compute_cut_cost(
    #         partitions=partitions,
    #         preference_graph=preference_graph,
    #         _default_weights=_default_weights,
    #         _weight_key=_weight_key)
    #     occupancy_cost = compute_occupancy_cost(
    #         partitions=partitions,
    #         _occupancy_costs=_occupancy_costs)
    #     return cut_cost, occupancy_cost  # must return a tuple
        
    toolbox.register("evaluate", evaluate,
                     preference_graph=preference_graph,
                     _default_weights=_default_weights,
                     _weight_key=_weight_key,
                     _occupancy_costs=_occupancy_costs)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # __name__ __main__ is a windows safety checksafety check on windows
    if __name__ == "__main__" and _num_workers > 0: 
        pool = mp.Pool(_num_workers)
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

        
    population = toolbox.population(population_size)
    hall_of_fame = tools.HallOfFame(_hall_of_fame_size)
    cost_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    size_stats = tools.Statistics(lambda ind: len([l for l in ind if len(l) > 0]))
    multistats = tools.MultiStatistics(cut_cost=cost_stats, parition_size=size_stats)
    multistats.register("avg", lambda x: np.round(np.mean(x), 3))
    multistats.register("std", lambda x: np.round(np.std(x), 3))
    multistats.register("min", lambda x: np.round(np.min(x), 3))
    multistats.register("max", lambda x: np.round(np.max(x), 3))
    logbook = tools.Logbook()
    logbook.header = "gen", "cut_cost", "parition_size"
    logbook.chapters["cut_cost"].header = "avg", "std", "min", "max"
    logbook.chapters["parition_size"].header = "avg", "std", "min", "max"
    
    if verbose:
        gen_iter = tqdm(range(n_generations), desc="Evolving")
    else:
        gen_iter = range(n_generations)

    # initially validate current population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    for gen_idx in gen_iter:
        hall_of_fame.update(population)  # move to beginning, before initial update
        record = multistats.compile(population)
        # hof_record = cost_stats.compile(hall_of_fame)
        logbook.record(gen=gen_idx, **record)
        if verbose:
            tqdm.write(logbook.stream)
        if timeLimit is not None:
            if time.time() > start_time + timeLimit:
                # Break due to timeout
                break
        selected = toolbox.select(population, population_size)
        # clone selected individuals to not overwrite them in the HoF
        offspring = list(toolbox.map(copy.deepcopy, selected))
        nums_to_move = np.clip(
            np.random.poisson(move_rate, population_size), 0, preference_graph.order())
        nums_to_shuffle = np.random.poisson(shuffle_rate, population_size)
        mutated_offspring = []
        for potential_mutant, num_to_move, num_to_shuffle in zip(
            offspring, nums_to_move, nums_to_shuffle):
            if num_to_move > 0 or num_to_shuffle > 0:
                potential_mutant = toolbox.mutate(
                    potential_mutant,
                    num_to_move=num_to_move,
                    num_to_shuffle=num_to_shuffle)
                del potential_mutant.fitness.values
            mutated_offspring.append(potential_mutant)
        unevaluated_pop = [ind for ind in mutated_offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, unevaluated_pop)
        for ind, fit in zip(unevaluated_pop, fitnesses):
            ind.fitness.values = fit
        population = mutated_offspring

    if _num_workers > 0: 
        pool.close()

    if _return_hof:
        return _return_hof
    return dict(enumerate(hall_of_fame[0])), hall_of_fame[0].fitness.values


if __name__ == "__main__":
    from pprint import pprint

    preference_graph = nx.karate_club_graph()

    initial_solution = initial_estimates(
        preference_graph=preference_graph,
        partition_sizes=20*[2] + 10*[3],
        max_used_partitions=20)
    print(initial_solution)

    # final_population, hall_of_fame = solve_ea(
    solution, cost = solve_ea(
        preference_graph,
        partition_sizes = 20*[2] + 10*[3],
        max_used_partitions = 20,
        _occupancy_costs = 20*[0.005] + 10*[0.006],
        n_generations = 100,
        population_size = 1000,
        # initial_solution=initial_solution,
        verbose=True,
        _num_workers=4,
        timeLimit=60)
    solution_nonempty = {k:v for k,v in solution.items() if len(v) > 0}
    print(f"{cost=}, {len(solution_nonempty)=}")
    pprint(solution_nonempty)
    print(preference_graph.order())
    print(sum(len(v) for v in solution.values()))
    print(len(set(sum(solution.values(), []))))