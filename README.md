# Room Matcher

Some python code to solve the [graph partition problem](https://en.wikipedia.org/wiki/Graph_partition) in particular with its application to assigning individuals to rooms based on their cohabitation preferences i.e. where vertices are individuals, edges are cohabitation preferences, and partitions are rooms.

## Limitations

The problem is set up so that individuals who want to be placed together will be, as far as possible. This does not mean that individuals who want to be placed *exclusively* together will be, as their is no component included for exclusivity. Exclusive room assignments need to be done manually and left out of the algorithm.

## Hard Problem

The graph partition problem is, in general, NP-hard. Hence, we can use an exact solver for the NP-hard integer program [PuLP](https://pypi.org/project/PuLP/) to find a solution if the graph is small enough. If the graph is too large, the solver may not finish before the heat death of the universe, and we may prefer to use [simulated annealing via SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) or an [evolutionary algorithm with DEAP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html).