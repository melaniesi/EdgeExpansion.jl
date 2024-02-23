# ==========================================================================
#   This file is part of EdgeExpansion
# --------------------------------------------------------------------------
#   Copyright (C) 2024 Melanie Siebenhofer <melaniesi@edu.aau.at>
#   EdgeExpansion is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   EdgeExpansion is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see https://www.gnu.org/licenses/.
# ==========================================================================

module PolytopeGraphGenerator


import Erdos
using HiGHS
using JuMP
using LinearAlgebra
import Random

export laplacian_rand01polytope, grlex, grevlex


#-----------------------------#
#    random 0/1-polytopes     #
#-----------------------------#

"""
    laplacian_rand01polytope(n, dim; seed=0)

Return the Laplacian matrix of the graph of a random
0/1-polytope with `n` vertices in dimension `dim`.

It is possible to provide a `seed`, the default value
of the seed is 0.
"""
function laplacian_rand01polytope(n, dim; seed=0)
    return laplacian_of_polytope_withoutPolymake(randompoints(n, dim, seed=seed))
end

"""
    randompoints(m, dim; polymake_input=false, seed=0)

Return a list of `m` 0/1-points of dimension `dim` without duplicates.

If polymake_input is false, the points are returned as rows of a matrix.
Otherwise there is a leading column with all ones added, the output is
then ready to be processed as input for Polymake's Polytope Constructor,
i.e., a matrix with `m` rows and `dim + 1` columns, where the first column
is all ones.
"""
function randompoints(m, dim; polymake_input=false, seed=0)
    Random.seed!(seed)
    if m == 0
        @warn "Number of points is 0, return Nothing."
        return Nothing
    elseif m > 2^dim
        @warn "Number of points is greater than 2^dim, return all possible 0/1 points."
        m = 2^dim
    end
    V = Random.bitrand(dim)
    while(size(V,2)) < m
        v = Random.bitrand(dim)
        if v ∉ eachcol(V)
            V = [V v]
        end
    end
    return polymake_input ? [ones(Int, m, 1) transpose(V)] : transpose(V)
end

"""
    laplacian_of_polytope_withoutPolymake(V)

Return the Laplacian matrix of the graph of the polytope given by
its vertices.

The vertices of the polytopes are provided as the rows
of the matrix `V`. The edges are determined by
solving Linear Programs with the HiGHS Solver.
"""
function laplacian_of_polytope_withoutPolymake(V)
    m = size(V, 1)
    dim = size(V, 2)
    L = zeros(Int64, m, m)
    for i = 1:m 
        for j = i+1:m
            # check whether there is an edge between vertex i and j
            # edge ij ∈ E(G) ⇔ dual problem infeasible
            # dual problem (to get certificate of infeasibility)
            dlp = Model(HiGHS.Optimizer)
            set_optimizer_attribute(dlp, "presolve", "on")
            #set_optimizer_attribute(dlp, "log_to_console", false)
            x = @variable(dlp, x[1:dim])
            y = @variable(dlp, y)
            z = @variable(dlp, z)
            @objective(dlp, Max, y - z)
            @constraint(dlp, V[i,:]' * x >= y)
            @constraint(dlp, V[j,:]' * x >= y)
            @constraint(dlp, V[vcat(1:i-1, i+1:j-1, j+1:m), :] * x .≤ z)
            optimize!(dlp);
            if primal_status(dlp) == INFEASIBILITY_CERTIFICATE || termination_status(dlp) == DUAL_INFEASIBLE
                L[i,j] = -1
                L[i,i] += 1
                L[j,j] += 1
            elseif termination_status(dlp) != OPTIMAL
                @warn "Termination status is neither optimal nor dual infeasible. Status $(termination_status(dlp))."
                @show primal_status(dlp)
                @show dual_status(dlp)
                @show termination_status(dlp)
            end
        end
    end
    return Symmetric(L, :U)
    #=
    # primal problem
    # --------------
        plp = Model(HiGHS.Optimizer)
        lambda = @variable(plp, 0 <= lambda[1:m])
        @constraint(plp, lambda[i] + lambda[j] == 1)
        @constraint(plp, sum(lambda[vcat(1:i-1, i+1:j-1, j+1:m)]) == 1)
        @constraint(plp, lambda[i] * V[i,:] + lambda[j] * V[j,:] == sum(lambda[k] * V[k,:] for k in vcat(1:i-1, i+1:j-1, j+1:m)))
        @objective(plp, Min, zeros(m)'*lambda)
        optimize!(plp)
        dual_status(plp)
        primal_status(plp)
        termination_status(plp)
    =#
end


#------------------------#
#    grlex polytopes     #
#------------------------#

"""
    grlex(d)

Return the Laplacian matrix of the graph of the grlex polytope in dimension `d`.

For details on the graph of the polytope,
see https://arxiv.org/pdf/1612.06332.pdf, figure 2.
The order of the vertices is θ, u3, ..., ud, w, v12, v13, v23, v14, ..., v1d, ..
.., v{d-1}d, 0

# Copyright
With permission, his code is based on a Matlab implementation written by Nicolo
Gusmeroli (2020) to get the adjacency matrix.

# Examples
```julia-repl
julia> print(grlex(5))
[5 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 0; -1 5 -1 -1 -1 0 0 -1 0 0 0 0 0 0 0 0;
-1 -1 6 -1 -1 -1 0 0 0 0 -1 0 0 0 0 0; -1 -1 -1 8 -1 -1 -1 -1 0 0 0 0 0 0 -1 0;
-1 -1 -1 -1 11 -1 -1 -1 -1 -1 -1 0 0 0 0 -1;
-1 0 -1 -1 -1 5 -1 0 0 0 0 0 0 0 0 0; 0 0 0 -1 -1 -1 5 -1 -1 0 0 0 0 0 0 0;
0 -1 0 -1 -1 0 -1 5 0 -1 0 0 0 0 0 0; 0 0 0 0 -1 0 -1 0 5 -1 -1 -1 0 0 0 0;
0 0 0 0 -1 0 0 -1 -1 5 -1 0 -1 0 0 0; 0 0 -1 0 -1 0 0 0 -1 -1 5 0 0 -1 0 0;
0 0 0 0 0 0 0 0 -1 0 0 5 -1 -1 -1 -1; 0 0 0 0 0 0 0 0 0 -1 0 -1 5 -1 -1 -1;
0 0 0 0 0 0 0 0 0 0 -1 -1 -1 5 -1 -1; 0 0 0 -1 0 0 0 0 0 0 0 -1 -1 -1 5 -1;
0 0 0 0 -1 0 0 0 0 0 0 -1 -1 -1 -1 5]
```
"""
function grlex(d)
    # graph from https://arxiv.org/pdf/1612.06332.pdf, figure 2
    v = [d, 1:d-1..., 1]
    p = length(v)
    n = sum(v)
    L = zeros(Int64,n,n)

    # clique between θ, u_k for 3 ⩽ k ⩽ d, w
    #               and v - blocks
    for i=1:p
        a1 = sum(v[1:i])
        a2 = a1 - v[i] + 1
        L[a2:a1,a2:a1] .= -1
    end

    # row-block 1:
    for r=1:d
        q = sum(v[1:r+1])
        L[r,q] = -1    # edge between theta,v_1,2; v_2,3, u_3, ..., v_d-1,d,u_d
                        # and between w and 0
        L[r,d+1:sum(v[1:r-1])] .= -1 # r ⩾ 3, edges of type (7) -> cliques
                                      # between v-cliques and u_k's
    end

    # row-blocks 2:d
    for i=2:p-2
        l = v[i]
        for k = 1:l
            L[sum(v[1:i-1])+k, sum(v[1:i])+k] = -1 # edges from clique v_i to
                                                    # clique v_i+1
        end
    end
    # row-block d+1
    r = sum(v[1:p-2]) + 1
    c = sum(v[1:p])
    L[r:r+v[p-1],c] .= -1 # edges from last v-clique to 0

    # make symmetric matrix! make laplacian!
    #L = UpperTriangular(L)
    L = Symmetric(L,:U)
    for i = 1:n
        L[i,i] -= sum(L[:,i])
    end
    return L
end



#--------------------------#
#    grevlex polytopes     #
#--------------------------#

"""
    grevlex(d)

Return the Laplacian matrix of the graph of the grevlex polytope
in dimension `d`.

See https://arxiv.org/pdf/1612.06332.pdf, figure 3 for a description
of the graph of the grevlex polytope in dimension `d`
The order of the vertices is u2, u3, ..., u{d+1}, θ, v13, ..., v1{d+1},
v24, ..., v2{d+1}, v35, ..., v{d-2}v{d+1}, v{d-1}{d+1}

# Copyright
With permission, his code is based on a Matlab implementation written by Nicolo
Gusmeroli (2020) to get the adjacency matrix.

# Examples
```julia-repl
julia> grevlex(4)
11×11 Symmetric{Int64,UpperTriangular{Int64,Array{Int64,2}}}:
  4  -1   0   0   0  -1  -1  -1   0   0   0
 -1   4  -1   0   0   0   0   0  -1  -1   0
  0  -1   4  -1   0  -1   0   0   0   0  -1
  0   0  -1   4  -1   0  -1   0  -1   0   0
  0   0   0  -1   4   0   0  -1   0  -1  -1
 -1   0  -1   0   0   4  -1  -1   0   0   0
 -1   0   0  -1   0  -1   5  -1  -1   0   0
 -1   0   0   0  -1  -1  -1   6   0  -1  -1
  0  -1   0  -1   0   0  -1   0   4  -1   0
  0  -1   0   0  -1   0   0  -1  -1   5  -1
  0   0  -1   0  -1   0   0  -1   0  -1   4
```
"""
function grevlex(d)
    # graph from https://arxiv.org/pdf/1612.06332.pdf, figure 3
    # order:  u2, ... , u{d+1}, 0,
    #         v13, ..., v1{d+1},
    #         v24, ... , v2{d+1},
    #         v35, ... v3{d+1},
    #         v{d-2}{d}, v{d-2}{d+1},
    #         v{d-1}{d+1}
    #   (row-wise)
    v = [d+1,d-1:-1:1 ...]
    n = (d^2 + d + 2) >> 1

    L = zeros(Int64, n,n)
    # row block 1
    # L[2:n+1:n*d] .= -1       # edges u_2-u_3-u_{d+1}-0
    L[n+1:n+1:n*(d+1)] .= -1 # edges u_2-u_3-u_{d+1}-0
    for i=2:d
        r = v[i]
        m = sum(v[1:i-1])
        L[d-r,m+1:m+r] .= -1 # (d-r) = index of u{i+1}, horizontal clique,
                              # edges between u{i+1} and each of the vertices
                              #               v{i}{i+2},...,v{i}{d+1}
        L[m*n+d-r+2:n+1:n*(m+r)] .= -1 # edge between v{i}{j} and u{j+1} for
                                        # all j with  i+2 ⩽ j ⩽ d+1
    end
    # other row blocks
    for j = 2:d
        r = v[j]
        m = sum(v[1:j-1])
        L[m+1:m+r,m+1:m+r] .= -1 # horizontal cliques (v{i}{i+2},...,v{i}{d+1})
        for i=j+1:d
            r2 = v[i]
            m2 = sum(v[1:i-1])
            L[m2*n+m+1+r-r2:n+1:n*(m2+r2)] .= -1 # edge to elem below in row of u{i}
                                                  # for vertical cliques
                                                  # edge betw. v{j-1}. and v{i-1}.
        end
    end
    # make symmetric matrix! make laplacian!
    #L = UpperTriangular(L)
    L = Symmetric(L,:U)
    for i = 1:n
        L[i,i] -= sum(L[:,i])
    end
    return L
end


end # module
