# EdgeExpansion.jl

Package to compute the edge expansion of an undirected graph $G$. The edge expansion of $G$ is defined as
```math
h(G) = \min_{\emptyset \neq S \subset V} \frac{\lvert \partial S \rvert}{\min \{ \lvert S \rvert, \lvert S \setminus V \rvert\}}
```
where $\lvert \partial S \rvert$ denotes the size of the cut induced by the set of vertices $S$, i.e.,
```math
\partial S = \{ \{i,j\} \in E(G) \mid i \in S, j \in V \setminus S\}.
```

### Installation
To enter the package mode press ```]``` and to exit press ```backspace```
```julia
pkg> add https://github.com/melaniesi/EdgeExpansion.jl.git
```
##### Dependencies
To use the package `EdgeExpansion.jl` the following solvers are needed. 
* [Mosek Aps](https://www.mosek.com/)
* [Gurobi](https://www.gurobi.com/)

If BiqBin is used as solver for the $k$-bisection problem, a modification of [BiqBin's source code](http://biqbin.eu/Home/Features#BiqBin) has to be installed. The modifications are tracked in the GitHub Repository [melaniesi/biqbin-modification-edgeexpansion](https://github.com/melaniesi/biqbin-modification-edgeexpansion).

### Example
```julia
julia> using EdgeExpansion
julia> import EdgeExpansion.PolytopeGraphGenerator
julia> biqbinpath = "/home/user/Code/biqbin-expedis/" # set path to BiqBin
julia> L = PolytopeGraphGenerator.grevlex(3)
7×7 LinearAlgebra.Symmetric{Int64, Matrix{Int64}}:
  3  -1   0   0  -1  -1   0
 -1   3  -1   0   0   0  -1
  0  -1   3  -1  -1   0   0
  0   0  -1   3   0  -1  -1
 -1   0  -1   0   3  -1   0
 -1   0   0  -1  -1   4  -1
  0  -1   0  -1   0  -1   3
julia> result_1 = split_and_bound(L, "grevlex-3", biqbin=true, biqbinpath=biqbinpath, ncores=4);
julia> result_2 = dinkelbach(L, "grevlex-3", biqbin_path=biqbin_path, ncores=4);
```

Further examples can be found in the folder [`examples/`](examples/) of this project.

### References
This package is part of the publications

Akshay Gupte, Melanie Siebenhofer, Angelika Wiegele. (2024). _Edge expansion of a graph: Exploring SDP-based computational strategies._ [Manuscript in preparation].

Akshay Gupte, Melanie Siebenhofer, Angelika Wiegele. (in press). _Computing the Edge Expansion of a Graph using Semidefinite Programming._ In: Combinatorial Optimization. ISCO 2024. Lecture Notes in Computer Science. Springer.