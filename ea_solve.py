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


def solve_ea(
    preference_graph: nx.DiGraph,
    partition_sizes: Sequence[int],  # all rooms and their sizes, for example 40*[2] + 10*[3] for 40 2-bedrooms and 10 3-bedrooms
    population_size: int,
    n_generations: int,
    move_rate: float=1.,
    max_used_partitions: int|None=None,  # if None, assume all partitions can be used
    verbose: bool=True,
    initial_solution: dict[int,str]|None=None,
    _occupancy_costs: Sequence[float]|float=0,
    _default_weights: Callable[[nx.DiGraph, tuple], float] = lambda g,e: 1.,
    _weight_key: str='weight',
    _hall_of_fame_size: int|None=None,
    _num_workers: int=0,
    timeLimit: float|None=None  # seconds
) -> tuple[dict[str,dict], tuple[float,float]]:
    if timeLimit is not None:
        start_time = time.time()
    
    if _hall_of_fame_size is None:
        _hall_of_fame_size = population_size
    
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
    toolbox.register("mutate", move_elements,
                     max_used_partitions=max_used_partitions,
                     partition_sizes=partition_sizes)
    # move_elements(
    #     partitions: list[list],
    #     partition_sizes: Sequence[int],
    #     max_used_partitions: int,
    #     num_to_move: int=1
    # )

    def _evaluate(partitions: list) -> float:
        cut_cost = compute_cut_cost(
            partitions=partitions,
            preference_graph=preference_graph,
            _default_weights=_default_weights,
            _weight_key=_weight_key)
        occupancy_cost = compute_occupancy_cost(
            partitions=partitions,
            _occupancy_costs=_occupancy_costs)
        return cut_cost, occupancy_cost  # must return a tuple
        
    toolbox.register("evaluate", _evaluate)
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
        mutated_offspring = []
        for potential_mutant, num_to_move in zip(offspring, nums_to_move):
            if num_to_move > 0:
                potential_mutant = toolbox.mutate(potential_mutant, num_to_move=num_to_move)
                del potential_mutant.fitness.values
            mutated_offspring.append(potential_mutant)
        unevaluated_pop = [ind for ind in mutated_offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, unevaluated_pop)
        for ind, fit in zip(unevaluated_pop, fitnesses):
            ind.fitness.values = fit
        population = mutated_offspring

    # return population, hall_of_fame
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
        population_size = 100,
        # initial_solution=initial_solution,
        verbose=True,
        timeLimit=60)
    solution_nonempty = {k:v for k,v in solution.items() if len(v) > 0}
    print(f"{cost=}, {len(solution_nonempty)=}")
    pprint(solution_nonempty)
    print(preference_graph.order)
    print(sum(len(v) for v in solution.values()))
    print(len(set(sum(solution.values(), []))))

    # print(
    #     "FINAL POPULATION",
    #     *zip((ind.fitness.values for ind in final_population), final_population),
    #     sep="\n")
    # print(
    #     "HALL OF FAME",
    #     *zip((ind.fitness.values for ind in hall_of_fame), hall_of_fame[:3]),
    #     sep="\n")