# Room Matcher

Some python code to solve the [graph partition problem](https://en.wikipedia.org/wiki/Graph_partition) in particular with its application to assigning individuals to rooms based on their cohabitation preferences i.e. where vertices are individuals, edges are cohabitation preferences, and partitions are rooms.

## Limitations

The problem is set up so that individuals who want to be placed together will be, as far as possible. This does not mean that individuals who want to be placed *exclusively* together will be, as their is no component included for exclusivity. Exclusive room assignments need to be done manually and left out of the algorithm.

## Hard Problem

The graph partition problem is, in general, NP-hard. Hence, we can use an exact solver for the NP-hard integer program [PuLP](https://pypi.org/project/PuLP/) to find a solution if the graph is small enough. If the graph is too large, the solver may not finish before the heat death of the universe, and we may prefer to use [simulated annealing via SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) or an [evolutionary algorithm with DEAP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html).

### Linear Integer Program

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