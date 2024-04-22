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

using JuMP
using LinearAlgebra
using Mosek
using MosekTools
using Random


"""
    compute_upperbounds_bestsolution(L)

Compute upper bounds on h_k(G) for 1 ≤ k ≤ ⌊n/2⌋.

The graph G is given by its Laplacian matrix `L`.
The bound computation is done with simulated annealing.
This function returns a vector of the upper bounds u_k
and the cut giving the smallest upper bound u* on the cheeger
constant h(G).
"""
function compute_upperbounds_bestsolution(L)
    Random.seed!(0)
    n = size(L, 1)
    nhalf = Int(floor(n / 2))

    # do computation on nthreads threads
    nthreads = Threads.nthreads()

    upperBounds = similar(1:nhalf, Float64)
    best_ubs = similar(1:nthreads, Float64)
    best_sols = [[] for _ in 1:nthreads]

    it_perthread = Int(ceil(nhalf / nthreads))
    shuffledindices = shuffle(1:nhalf)
    threadhandles = []
    for i in 1:nthreads
        istart = (i - 1) * it_perthread + 1
        iend = min(i * it_perthread, nhalf)
        t = Threads.@spawn update_upper_bounds!(upperBounds, best_ubs, best_sols, L, shuffledindices[istart:iend], i)
        push!(threadhandles, t)
    end

    for t in threadhandles
        wait(t)
    end

    return (upperBounds, best_sols[argmin(best_ubs)])
end

"""
    update_upper_bounds!(upperBounds, best_ubs, best_sols, L, indices, threadid)

Function for thread number `threadid` to compute the upper bounds for [`compute_upperbounds_bestsolution`](@ref).

Thread `threadid` works on the values `indices` for k and stores
the smallest upper bound in `best_ubs[threadid]` and the best solution vector
in `best_sols[threadid]`.
In `upperBounds[indices]` it stores the computed upper bounds.
"""
function update_upper_bounds!(upperBounds, best_ubs, best_sols, L, indices, threadid)
    opt = []
    best_ub = Inf
    for k in indices
         ub, perm = kCut_simulatedAnnealing(L,k,trials=10,locSearch=true)
         upperBounds[k] = ub / k
         if upperBounds[k] < best_ub
            best_ub = upperBounds[k]
            opt = perm
         end
        println("upper bound for k=",k," is ",upperBounds[k])
    end
    best_ubs[threadid] = best_ub
    best_sols[threadid] = opt
end


"""
    compute_lower_bounds_simpleSDP(L)

Compute lower bounds on h_k(G) for 1 ≤ k ≤ ⌊n/2⌋.

Computes lower bounds coming from the basic SDP relaxation
with Mosek and returns a vector of the lower bounds l_k on h(G).
"""
function compute_lower_bounds_simpleSDP(L)
    n = size(L,1)
    np1 = n+1
    lowerBounds = Vector{Float64}()
    for k = 1:floor(n/2)
        model = Model(Mosek.Optimizer)
        @variable(model, Y[1:np1,1:np1], PSD)
        @constraint(model, LinearAlgebra.diag(Y[2:np1,2:np1]) .== Y[1,2:np1])
        @constraint(model, Y[1,1] == 1)
        @constraint(model, 1/2*sum(Y[1,2:np1]) + 1/2*sum(Y[2:np1,1]) == k)
        @constraint(model, LinearAlgebra.dot(ones(Int64,n,n),Y[2:np1,2:np1]) == k^2)

        @objective(model, Min, LinearAlgebra.dot(L, Y[2:np1,2:np1]))

        optimize!(model)
        val = objective_value(model)
        val = val - floor(val) < 10e-4 ? val : ceil(val)
        push!(lowerBounds,val/k)
    end
    return lowerBounds
end
