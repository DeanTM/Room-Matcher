# Room Matcher

Some python code to solve the [graph partition problem](https://en.wikipedia.org/wiki/Graph_partition) in particular with its application to assigning individuals to rooms based on their cohabitation preferences i.e. where vertices are individuals, edges are cohabitation preferences, and partitions are rooms.

## Specification

We assume that we have a range of possible maximal partition sizes, but that no more than a subset of them can be used. A specific use case is when we have $N$ rooms available, but only $n < N$ rooms can have extra people in them. Then we have a collection of $N + n$ possible maximal sizes, of which only $N$ can be used.

The preferences individuals have to being in the same room as another individual is encoded in a networkx digraph, where edges reflect preferences, and edge weights reflect the importance of those preferences. As such, each individual needs a unique identifier (like an email) to correspond to a unique node in the digraph.

## Limitations

The problem is set up so that individuals who want to be placed together will be, as far as possible. This does not mean that individuals who want to be placed *exclusively* together will be, as their is no component included for exclusivity. Exclusive room assignments need to be done manually and left out of the algorithm.

## Contents

* [initialise.py](./initialise.py) contains two helper functions to find initial candidate solutions:
  * `initial_random` fairly randomly allocates individuals to sets in the partition (with a bias towards using the larger sets, and allocating first the individuals from larger connected components, because it was easier to respect restrictions that way). This is the default initialisation used in the evolutionary algorithm if no other initial solution is given.

  * `initials_estimates` recursively uses the Fielder vector to compute sparse cuts of the preferences graph, before assigning groups are vertices to sets within the partition. In general this should perform better than random.

* [mip_solve.py](./mip_solve.py) implements the mixed integer program in PuLP and uses its default solver to find a solution. This is implemented in the function `solve_mip` which can be seen in the `if __name__ == "__main__"` block at the end of the file.

* [ea_solve.py](./ea_solve.py) implements the simple evolutionary algorithm to solve the partition problem. This is implemented in the function `solve_ea` which can be seen in the `if __name__ == "__main__"` block at the end of the file.

* [main.py](./main.py) solves the problem by finding initial estimates, then running `solve_mip` and finally potentially improving the solution by running `solve_ea` on a dataset given as a tsv wherein nodes are specified as emails in column 0 and targets of outgoing edges are specified as emails in column 3 (for the particular use case of the 2024 Student's Retreat dataset).

## A Hard Problem

The graph partition problem is, in general, NP-hard. Hence, we can use an exact solver for the NP-hard integer program [PuLP](https://pypi.org/project/PuLP/) to find a solution if the graph is small enough. If the graph is too large, the solver may not finish before the heat death of the universe, and we may prefer to use [simulated annealing via SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) or an [evolutionary algorithm with DEAP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html).

### [Mixed Integer Program](./mip_solve.py)

We formulate it as follows. We have a set $V$ of vertices and a set $E \subset V \times V$ of directed edges. We have a set of edge weights $w_e \, \forall e\in E$ and a maximum number of used partitions $K$ of the available $N$. Each partition $i \in \underline{N} = \{1,\dots, N\}$ has a maximum capacity $C_i$. Furthermore, each partition $i$ has a cost $U_i$ (which may be $0$ for all $i$, as the user wishes). We want to find

$$
\begin{aligned}
&\argmin_{X \in \{0,1\}^{V\times \underline{N}}}  \sum_e w_e \cdot E_e + \sum_i U_i \cdot Y_i\\
&\text{where}\\
&\quad X_{v,i}, Y_i, E_e \in \{0,1\} \quad \forall v \in V \, \forall i \in \underline{N}, \forall e \in E\\
&\quad \sum_i X_{v,i} = 1 \quad \forall v \in V\\
&\quad \sum_v X_{v,i} \leq  C_i\quad \forall i \in \underline{N}\\
&\quad E_{(u,v)} \geq (X_{u,i} - X_{v,i}) \quad \forall i \in \underline{N} \, \forall (u,v) \in E\\
&\quad \sum_v X_{v,i} \leq |V| \cdot Y_i \quad \forall i \in \underline{N}\\
&\quad \sum_v X_{v,i} \geq Y_i \quad \forall i \in \underline{N}\\
&\quad \sum_i Y_i \leq K\\
\end{aligned}
$$

Going through line by line, the first line is our optimisation criterion. In it, $w_e$ is the weight of edge $e$, or equivalently the cost incurred by cutting it (having the different ends of $e$ being in different partitions), and $U_i$ is the cost of occupying partition $i$.

We define $X$, $Y$, and $E$ variables (indexed over the vertices of the graph, the available partitions, and the edges of the graph) which are, respectively, the assignment-to-partition variable, the partition-is-occupied variable, and the edge-is-cut variable.

The remaining lines constrain these variables to represent what we desire:

* each vertex $v$ must be assigned to exactly one partition $i$
* each partition $i$ must have fewer than $C_i$ vertices assigned to it
* each edge-is-cut variable $E_e$ where $e=(u,v)$ must be $1$ if the edge is cut i.e. if $X_{u,i} \neq X_{v,i}$ for some partition $i$
* the partition-is-occupied $Y_i$ variable must be 1 if there is any vertex in its partition $i$
* the partition-is-occupied $Y_i$ variable must be 0 if there is no vertex in partition $i$
* the number of occupied partitions must be less than our threshold $K$.

### [Evolutionary Algorithm](./ea_solve.py)

We can also solve the task by implementing an evolutionary algorithm (EA). One such simple one is done here. The EA mutates candidate partitions by moving a Poisson-sampled number of individuals between different sets in the partition, subject to the constraint that the number of occupied sets does not go above a specified threshold ($K$ above). Different solutions are then compared, and the best solutions are used to populate a new generation of answers.