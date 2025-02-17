[![DOI](https://zenodo.org/badge/750878738.svg)](https://zenodo.org/doi/10.5281/zenodo.13379324)
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

If BiqBin is used as solver for the $k$-bisection problem, version 1.1.0 of [BiqBin's source code](http://biqbin.eu/Home/Features#BiqBin) has to be installed. The modifications are tracked in the Git Repository [gitlab.aau.at/BiqBin/biqbin](https://gitlab.aau.at/BiqBin/biqbin). To run BiqBin, an installation of [mpich](https://www.mpich.org/) is required.

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

Akshay Gupte, Melanie Siebenhofer, Angelika Wiegele. (2024). _Edge expansion of a graph: SDP-based computational strategies._ [Manuscript submitted for publication].

Akshay Gupte, Melanie Siebenhofer, Angelika Wiegele. (2024). _Computing the Edge Expansion of a Graph using Semidefinite Programming._ In: Combinatorial Optimization. ISCO 2024. Lecture Notes in Computer Science. Springer. https://doi.org/10.1007/978-3-031-60924-4_9
