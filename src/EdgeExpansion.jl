# ==========================================================================
#   EdgeExpansion -- Program to compute the edge expansion of a graph
# --------------------------------------------------------------------------
#   Copyright (C) 2024 Melanie Siebenhofer <melaniesi@edu.aau.at>
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see https://www.gnu.org/licenses/.
# ==========================================================================

module EdgeExpansion

include("GraphBisection.jl")
using .GraphBisection
include("RudyGraphIO.jl")
include("PolytopeGraphGenerator.jl")

using Gurobi
using JuMP
using Random

export gurobi_solve_exact, split_and_bound, dinkelbach

#----------------------#
#     G U R O B I      #
#----------------------#

"""
    gurobi_solve_exact(L; lb=missing, timelimit=10800.0, nrthreads=4)

Compute the edge expansion of a graph with Gurobi.

Computes the edge expansion of a graph
given by its Laplacian matrix `L` with Gurobi.

# Arguments
- `lb=missing`: provides a lower bound on the edge expansion
                Note, that for Gurobi (version 11.0) it
                is beneficial to set `lb = 0`.
- `timelimit=10800`: time limit in seconds for Gurobi to solve
                     the optimization problem
- `nrthreads=4`:    number of threads available to Gurobi

Returns the edge expansion (or lower bound), the
relative gap obtained by Gurobi and the solve time (in seconds).
needed for Gurobi to solve the problem.
"""
function gurobi_solve_exact(L; lb=missing, timelimit=10800.0, nrthreads=4)
    n = size(L,1)
    m = Model(Gurobi.Optimizer)
    # MOI.set(m, MOI.AbsoluteGapTolerance(), 1e-6)
    set_time_limit_sec(m, timelimit)
    set_optimizer_attribute(m, "Threads", nrthreads)
    @variable(m, x[1:n], Bin) # binary variables x₁,…,xₙ
    if ismissing(lb)
        @variable(m, y)
    else
        @variable(m, y >= lb)
    end
    @objective(m, Min, y)
    @constraint(m, 1 <= sum(x[i] for i = 1:n) <= floor(n/2))
    @constraint(m, x'*L*x - y*sum(x[i] for i=1:n) <= 0)
    optimize!(m)
    return objective_bound(m), relative_gap(m), solve_time(m)
end

#----------------------------------#
#     S P L I T  &  B O U N D      #
#----------------------------------#

include("boundcomputation.jl")

"""
    split_and_bound(L, instancename="splitandbound-instance"; <keyword arguments>)

Compute the edge expansion of a graph with the split and bound algorithm.

The graph is given by its Laplacian matrix `L` and its name `instancename`.

# Arguments
- `biqbin=true`: uses biqbin to solve the k-bisection problem if it is true and
                 the ADMM branch-and-bound solver otherwise.
- `biqbin_path=missing`: if `biqbin=true` a path to the biqbin installation, e.g.,
                         "/home/user/Code/biqbin-expedis/", has to be provided.
                         The source code of biqbin can be downloaded from
                         https://gitlab.aau.at/BiqBin/biqbin.
- `ncores=4`: the number of cores to run biqbin on

Returns a solution dictionary with the entries
 - EdgeExpansion
 - OptimalCut
 - UpperBounds: a vector of the upper bounds u_k
 - CheapLowerBounds: a vector of the lower bounds l_k
 - IndicesBetterBound: indices k for which we used BiqBin/branch-and-bound to
                       obtain better lower bounds/or computed the optimum h_k
 - NewBoundsBiqBin: a vector with lower bounds, improved lower bounds or if needed optimal
                    values h_k
 - RuntimeNumerics: Dictionary with entries
      - TimePreprocessing: time needed for preprocessing in seconds
      - TimesBiqBin: vector of length n with the time spent to compute better lower bound/compute h_k
                     in position k
      - NrBBNodes: vector of length n with the number of branch-and-bound nodes needed to compute
                   better lower bound/compute h_k
"""
function split_and_bound(L, instancename="splitandbound-instance"; biqbin=true, biqbin_path=missing, ncores=4)
    Random.seed!(0)
    if biqbin && ismissing(biqbin_path)
        @warn "Path to BiqBin is missing but has to be provided. Use branch-and-bound for bisection instead."
        biqbin = false
    end

    # simple upper and lower bounds
    time_preproc = 0
    time_preproc += @elapsed upperBounds, opt = compute_upperbounds_bestsolution(L)
    time_preproc += @elapsed lowerBounds = compute_lower_bounds_simpleSDP(L)
    best_k = length(opt)
    global_upper_bound = upperBounds[best_k]

    indices_better_bound = findall(<(global_upper_bound), lowerBounds)
    new_upperBounds = copy(upperBounds)
    # run simulated annealing again with more trials
    # to maybe improve global upper bound
    improved_ub = false
    for k in indices_better_bound
        time_preproc += @elapsed ub, perm = kCut_simulatedAnnealing(L, k; trials=30, locSearch=true)
        new_ub = ub / k
        if new_ub < global_upper_bound
            improved_ub = true
            global_upper_bound = new_ub
            best_k = k
            opt = perm
            new_upperBounds[k] = new_ub
        end
    end
    if improved_ub
        indices_better_bound = findall(<(global_upper_bound), lowerBounds)
    end
    if isempty(indices_better_bound)
        # no lower bound below ub → we know the optimum
        # global_upper_bound = new_upperBounds[best_k] == lowerBounds[best_k]
        return Dict("EdgeExpansion" => global_upper_bound, "OptimalCut" => opt, "UpperBounds" => new_upperBounds, "CheapLowerBounds" => lowerBounds,
                    "NewBoundsBiqBin" => missing, "IndicesBetterBound" => missing,
                    "RuntimeNumerics" => Dict(zip(["TimePreprocessing", "TimesBiqBin", "NrBBNodes"],[time_preproc, missing, missing])) )
    end

    # we want to start with the most promising k
    # also in regards of potentially finding a better solution
    # (pushing down the global upper bound)
    sorted_UB_openIndices = sort(collect(zip(new_upperBounds[indices_better_bound], indices_better_bound)))
    indices_better_bound = getindex.(sorted_UB_openIndices, 2)
    new_lowerBounds = copy(lowerBounds)
    nrbbnodes = zeros(length(lowerBounds))
    times = zeros(length(lowerBounds))
    for k in indices_better_bound
        println("SOLVE BISECTION PROBLEM FOR k = $k")
        # solve bisection problem → multiply with k
        ub = ceil(global_upper_bound * k) # artificial ub to stop admm after lb/k ≥ global ub
        ub_k = Int(round(new_upperBounds[k] * k))
        lb = lowerBounds[k] * k
        if biqbin
            opt_k, sol_k, bbnodes, time_exact = bisection_biqbin_withUB(L, k, instancename, ub, ub_k, ncores, biqbin_path)
        else
            time_exact = @elapsed opt_k, sol_k, bbnodes = bisection_branchandbound(L, k, ub, lb)
        end
        new_lowerBounds[k] = sol_k / k
        nrbbnodes[k] = bbnodes
        times[k] += time_exact
        if !ismissing(opt_k) # found better solution than provided by ub 
            @assert sol_k / k <= global_upper_bound
            global_upper_bound = sol_k / k
            opt = opt_k
        end
    end

    return Dict("EdgeExpansion" => global_upper_bound, "OptimalCut" => opt, "UpperBounds" => new_upperBounds, "CheapLowerBounds" => lowerBounds,
                     "NewBoundsBiqBin" => new_lowerBounds, "IndicesBetterBound" => indices_better_bound,
                     "RuntimeNumerics" => Dict(zip(["TimePreprocessing", "TimesBiqBin", "NrBBNodes"],[time_preproc, times, nrbbnodes])) )

end

#-------------------------------#
#     D I N K E L B A C H       #
#-------------------------------#

include("Dinkelbach.jl")

"""
    dinkelbach(L, instancename="dinkelbach-instance"; <keyword arguments>)

Compute the edge expansion of a graph with an algorithm based on Dinkelbach.

The graph is given by its Laplacian matrix `L` and its name `instancename`.

# Arguments
- `biqbin_path=missing`: a path to the biqbin installation, e.g.,
                         "/home/user/Code/biqbin-expedis/", has to be provided.
                         The source code of biqbin can be downloaded from www.biqbin.eu
- `ncores=4`: the number of cores to run biqbin on

Returns a dictionary with the entries
 - OptimalCut: one side of the optimal Cheeger cut
 - time_ub:      time in seconds needed to compute upper bounds u_k (preprocessing)
 - guesses:      list of the guesses \$\\frac{\\gamma_n}{\\gamma_d}\$
 - numerators:   list of the numerators of the guesses \$\\gamma_n \$
 - denominators: list of he denominators of the guesses \$\\gamma_d \$
 - bb-time:      list of of times in seconds the branch-and-bound algorithm needed
                 to check guesses
 - bb-nodes:     list of branch-and-bound nodes needed to check guesses
"""
function dinkelbach(L, instancename="dinkelbach-instance"; biqbin_path=missing, ncores=4)
    if ismissing(biqbin_path)
        @error "Path to BiqBin must be provided."
    end
    time_ub=@elapsed upperBounds, opt = compute_upperbounds_bestsolution(L)
    global_ub, den = findmin(upperBounds)
    num = Int64(round(global_ub * den))
    cheeger, xopt, res_info = dinkelbach_maxcut(L, num // den, instancename, biqbin_path, ncores)
    if !ismissing(xopt)
        opt = findall(==(1), xopt)
    end
    res_info["OptimalCut"] = opt
    res_info["time_ub"] = time_ub
    return res_info
end

end # module
