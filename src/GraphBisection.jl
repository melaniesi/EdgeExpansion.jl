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

module GraphBisection

using DataStructures
using Dates
using Dictionaries
using Graphs
using Gurobi
using JSON
using JuMP
using LinearAlgebra
using MKL
using Printf
using Random
using SparseArrays

export bisection_branchandbound, bisection_biqbin_withUB, kCut_simulatedAnnealing   

# set constant Gurobi environment
const GRB_ENV = Ref{Gurobi.Env}()
function __init__()
    const GRB_ENV[] = Gurobi.Env()
    return
end

#----------------------#
#      B I Q B I N     #
#----------------------#

"""
    bisection_biqbin_withUB(L, k, graphname, ub, ub_bisection_k = missing, ncores=4, biqbinpath=missing)

Solve the `k`-bisection problem for the given graph with its Laplacian
matrix `L` with BiqBin's max-cut solver [[5]](#5).

# Arguments
- `graphname::String`: the name of the instance
- `ub`: (artificial) upper bound on the `k`-bisection problem (to stop algorithm earlier if needed)
- `ub_bisection_k`: upper bound on the `k`-bisection problem to formulate the max-cut problem
- `ncores=4`: number of cores to run the BiqBin algorithm on
- `biqbinpath=missing`: path to the BiqBin installation, e.g.: "/home/user/Code/biqbin-expedis/".

Returns the optimal cut (if it was possible to improve upon `ub`, otherwise it is `missing`),
the size of the cut, the number of branch-and-bound nodes and the
computation time in seconds BiqBin needed.   

# References
<a id="5">[5]</a> 
Nicolò Gusmeroli, Timotej Hrga, Borut Lužar, Janez Povh, Melanie Siebenhofer, and Angelika Wiegele (2022).
BiqBin: A Parallel Branch-and-bound Solver for Binary Quadratic Problems with Linear Constraints.
ACM Trans. Math. Softw. 48, 2.
"""
function bisection_biqbin_withUB(L, k, graphname::String, ub, ub_bisection_k = missing, ncores::Int=4, biqbinpath::String=missing)
    if ismissing(biqbinpath)
        throw(ErrorException("Path to BiqBin not provided."))
    end
    instancespath = biqbinpath*"Instances/"
    if !isdir(instancespath)
        mkdir(instancespath)
    end
    mpirunexe = try chomp(read(`which mpirun`, String)) #"/usr/bin/mpirun"
    catch
        throw(ErrorException("Can not find mpirun, please install mpich to run BiqBin."))
    end

    # write max cut input
    filepath = instancespath*graphname*"k$k.dat"
    offset = write_max_cut_input(L, k, filepath, ub_bisection_k)

    # solve max cut problem
    result = read(`$mpirunexe -n $ncores $(biqbinpath)biqbin $filepath $(biqbinpath)params Julia 5 30 $(offset - ub)`, String)
    result = JSON.parse(result)["ExecutionMD"]

    # retrieve solution of bisection problem
    # compute solution bisection problem (offset - solution max cut)
    cutsize = offset - result["Solution"]
    # in OneSideOfTheCut all vertices with assinged value 1
    # first vertex is helper for linear term and has to be 1
    # -> check & extract solution if solution is better as provided upper bound
    cutset = missing
    if cutsize < ub
        # found better solution
        cutset = result["OneSideOfTheCut"] .- 1
        if 0 ∈ cutset
            deleteat!(cutset, findfirst(iszero, cutset))
        else
            cutset = setdiff(1:size(L,1), cutset)
        end
    end
    foreach(rm, filter(startswith(filepath*"_Julia_"), readdir(instancespath,join=true)))

    # delete temporary output of biqbin
    return (cutset, cutsize, result["BabNodes"], result["ExecutionTime"])
end

"""
    write_max_cut_input(L,k)

Write max-cut input in `filepath` to compute
the `k`-bisection divided by the constant `k`.

Transforms the binary quadratic program with
linear equality constraint to compute the
`k`-bisection where G is given by its Laplacian matrix `L`
into a max cut problem and writes the
corresponding max-cut input file (rudy format)
in `filepath` and returns the `offset` such that
`k`-bisection(G) = `offset` - max cut solution.

For the transformation an upper bound on the
`k`-bisection problem needs to be computed.
It can optionally be provided by `ub_bisection_k`.
"""
function write_max_cut_input(L, k, filepath, ub_bisection_k = missing)
    n = size(L, 1)
    if ismissing(ub_bisection_k)
        ub_bisection_k = kCut_simulatedAnnealing(L, k)[1]
    end
    σ = ub_bisection_k + 1/4
    io = open(filepath, "w")
    println(io, "$(n+1) $(Int(n*(n+1)/2))")
    entry1 = -4*σ*(2*k - n)
    for j=2:n+1
        println(io, "1 $j $entry1")
    end
    for i = 1:n
        for j = i+1:n
            println(io, "$(i+1) $(j+1) $(L[i,j] + 4*σ)")
        end
    end
    close(io)
    return Int(4*σ)*(k-n)^2
end

#---------------------------#
#      A D M M  - B & B     #
#---------------------------#

mutable struct BBNode
    depth::Int64                           # in BB tree
    lb::Float64                            # lower bound
    vertices::Vector{Union{Missing, Bool}} # assignment of vertices
    k::Int64
    estimated_improvement::Float64         # estimated improvement in ADMM by adding violated triangles
end

struct BQPIneq
    indices::Tuple{Int64, Int64, Int64} # (i,j,l) with i < j
    lstar::Int64
    #type::Int64
end

"""
    bisection_branchandbound(L, k, ub=Inf, lb=-Inf; <keyword arguments>)

Branch-and-bound algorithm to compute the `k`-bisection
of a graph given by its Laplacian matrix `L`.

An initial upper bound `ub` and lower bound `lb` on the
bisection problem can be provided.
Returns `(opt, ub, counter_bbnodes)` where `opt = missing`
if the provided initial upper bound was the optimum, so
no better `k`-bisection was found.

The lower bounding procedure is based on the algorithm
introduced in [[3]](#3).

### Details on the algorithm:
* BFS (best first search), branching rule most fractional
* prune a node if ub - lb < 1 - 1e-6
* Parameters for heuristic: 20 trials, local search
* Parameters for ADMM: max_new_bqpineq = 3*size(L)

# Arguments
- `admm_epstols=(1e-3,5e-3,1e-3)`: see also [`admm_bisect!`](@ref).
- `admm_maxouterloops=5`: maximum outer loops (adding violated triangle inequalities)

# References
<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function bisection_branchandbound(L, k, ub=Inf, lb=-Inf; admm_epstols=(1e-3,5e-3,1e-3), admm_maxouterloops=5)
    Random.seed!(0)
    # initialization
    n = size(L, 2)

    xvec = Vector{Union{Missing, Bool}}([missing for _ in 1:n])
    bb_node = BBNode(1, lb, xvec, k, Inf)
    open_nodes = PriorityQueue{BBNode, Float64}(Base.Order.Reverse)
    enqueue!(open_nodes, bb_node, 0.0)

    opt = missing
    counter_bbnodes = 0


    while !isempty(open_nodes)
        bb_node = dequeue!(open_nodes); counter_bbnodes += 1;
        local_vertices = findall(ismissing, bb_node.vertices)
 
        local_L, linear_term, const_term = shrink_L(bb_node.vertices, L)
        #const_term = bb_node.const_term

        println("======================================================================")
        println("BRANCH & BOUND NODE #$(counter_bbnodes) (still open: $(length(open_nodes)))")
        println("======================================================================")
        @printf("depth: %18d\n", bb_node.depth)
        @printf("vertices (sub)graph: %4d\n", length(local_vertices))
        @printf("constant term: %10d\n", const_term)
        @printf("lower bound: %17.4f\n", bb_node.lb)
        @printf("upper bound: %12d\n", ub)
                

        if ub - bb_node.lb < 1 - 1e-6
            println("----------------------------------------------------------------------")
            println("PRUNE!\n")
            continue
        end
        println("======================================================================")
        new_ub, perm = kCut_simulatedAnnealing(local_L + diagm(linear_term), bb_node.k, trials=20, locSearch=true)
        new_ub += const_term
        println("upper bound for subproblem: $new_ub")
        
        if new_ub < ub
            ub = new_ub
            @printf("NEW UPPER BOUND: %7d\n", new_ub)
            opt = [local_vertices[perm]; findall(isequal(1), bb_node.vertices)]
            if ub - bb_node.lb < 1 - 1e-6
                println("----------------------------------------------------------------------")
                println("PRUNE!\n")
                continue
            end 
        end
        println("\ncompute new lower bound with ADMM\n")# (adjusted ub: $(ub - const_term))\n")
        println("----------------------------------------------------------------------")
        new_lb, Xopt = admm_bisect!(bb_node, local_L, linear_term, const_term, 3 * size(local_L,1),
                                    ub, eps_tols=admm_epstols, max_outerloops=admm_maxouterloops)
        println("\n----------------------------------------------------------------------")
        @printf("NEW LOWER BOUND: %11.4f\t(upper bound: %6d)\n", new_lb, ub)
        println("----------------------------------------------------------------------\n")
        if new_lb > bb_node.lb
            bb_node.lb = new_lb
        end
        if ub - bb_node.lb > 1 - 1e-6
            # branch!
            local_vert = branchingDecision_mostfrac(Xopt)
            #local_vert = branchingDecision_closest2one(Xopt)
            global_vert = local_vertices[local_vert]
            println("BRANCH on vertex $(global_vert)\n")
            enqueue!(open_nodes, branch_on_vertex(bb_node, global_vert, 1), ub - bb_node.lb)
            enqueue!(open_nodes, branch_on_vertex(bb_node, global_vert, 0), ub - bb_node.lb)
        else
            println("PRUNE!\n")
        end
    end
    println("======================================================================")
    if ismissing(opt)
        println("Provided upper bound $ub was ≤ the optimum")
    else
        sort(opt)
        println("Optimum: $ub")
        println("Solution: $opt")
        
    end
    println("number of b&b nodes: $counter_bbnodes")
    println("======================================================================")

    return (opt, ub, counter_bbnodes) # opt is missing if ub was already the optimum
end

"""
    shrink_L(v, L)

Shrink the Laplacian matrix `L` for the bisection
problem if the partial assignment `v` of the
vertices is given.

The vector `v` has dimension `n` and on position
i it is true if vertex i is set to 1 and false if
it is set to 0 and missing if vertex i has no
assignment yet (hence is a local vertex in the
subproblem of the bisection problem).
Returns `(local_L, linear_term, const_term)`.
The objective x'Lx with the assignment of x as given
in `v` is then equivalent to
`y' * local_L * y + linear_term' * y + const_term`.
"""
function shrink_L(v, L)
    local_v = findall(ismissing, v)
    in_S = findall(x -> !ismissing(x) && x, v)
    notin_S = findall(x -> !ismissing(x) && !x, v)
    local_L = L[local_v,local_v]
    d = diag(L)
    lt = zeros(length(v))
    for i in in_S
        d += L[:,i]
        lt += L[:,i]
    end
    for i in notin_S
        d += L[:,i]
        lt -= L[:,i]
    end
    d = d[local_v]
    lt = lt[local_v]
    for i in 1:length(local_v)
        local_L[i,i] = d[i]
    end
    const_term = sum(L[in_S,in_S])
    return(local_L, lt, const_term)
end

"""
    kCut_simulatedAnnealing(L, k; <keyword arguments>)

Heuristic (simulated annealing) for the min `k`-bisection problem.

# Arguments
 - `trials=10`: the number of restarts
 - `locSearch=false`: if `true`, a local search is performed
                to find local improvements after each outer iteration,
                i.e., before restart with new random permutation.

# Copyright
With permission, this code is based on Elisabeth Gaar's
C implementation of simulated annealing heuristic for the QAP,
as used in [[4]](#4). The simulated annealing heuristic was
introduced by Burkard and Rendl in [[2]](#2).

<a id="4">[4]</a> 
Gaar, E. (2018). Efficient Implementation of SDP Relaxations for the Stable Set Problem.
Ph.D. thesis, Alpen-Adria-Universität Klagenfurt.

<a id="2">[2]</a> 
Rainer E. Burkard and Franz Rendl. “A thermodynamically motivated simulation
procedure for combinatorial optimization problems”. In: European Journal of
Operational Research 17 (1984), pp. 169–174.

"""
function kCut_simulatedAnnealing(L, k; trials=10, locSearch=false)
    n = size(L,1)
    # SIMULATED ANNEALING
    # for QAP with H = ones(k,k)
    # we want to
    # minimize < L(optimal_permutation,optimal_permutation), H>
    # which is equivalent to
    # minimize cut(S,S') s.t. |S| = k

    # parameters for sim an
    miter = 1*n # inner iterations miter*n
    ft = 0.7 # factor to decrease temperature, < 1, (0.6)
    fiter = 1.15 # factor to increase inner trials (1.1)

    # compute initial temp
    # t = ( sum(sum(abs(H))) * sum( sum( abs(X))))  / n / (n-1)
    t = k^2 * sum(abs.(L)) / (n * (n-1)) # sum(abs.(H)) * sum(abs.(L)) / (n * (n-1))
    bestFoundCost = Base.Inf64
    bestFoundPerm = Vector(1:n)
    for nrtrial = 1:trials
        # random permutation
        permutation = Vector(1:n)
        shuffle!(permutation);
        
        # initialize current values
        t1 = t
        m1 = miter
        curBestSol = sum(L[permutation[1:k],permutation[1:k]])
        changeOfSol = 0

        done = false
        while !done
            done = true
            # do m1 iterations at constant temperature
            for nr_it_t = 1:m1
                # random variables i1 in (1,...k) and i2 in (k+1,...,n)
                i1 = rand(1:k)
                i2 = rand(k+1:n)
                
                pi1 = permutation[i1]
                pi2 = permutation[i2]
                changeOfSol = 2 * sum(-L[pi1,permutation[j]] + L[pi2,permutation[j]] for j = vcat(1:i1-1,i1+1:k); init = 0) - L[pi1,pi1] + L[pi2,pi2]

                # accept swap?
                accept = false
                if changeOfSol > 0 # accept worse sol with certain probability if it is not too worse
                    dt1 = changeOfSol/t1
                    if dt1 <= 5 && rand() < exp(-dt1) # method by Kirkpatrick et al.
                        accept = true
                    end
                else
                    accept = true
                end
                if accept
                    if abs(changeOfSol) > 0.001
                        done = false
                    end
                    permutation[i2], permutation[i1] = permutation[i1], permutation[i2]
                    curBestSol += changeOfSol
                    if curBestSol < bestFoundCost
                        bestFoundCost = curBestSol
                        bestFoundPerm = copy(permutation)
                    end
                end             
            end # m1 iterations at constant temperature
            t1 *= ft
            m1 = round(m1*fiter)
        end # iterations over temperature t1
        if locSearch
            curBestSol = findLocalOptimizer!(bestFoundPerm,L,k)
        end
    end # iteration for one starting permutation

    return(bestFoundCost, bestFoundPerm[1:k])
end

"""
    findLocalOptimizer!(L, k, permutation)

Find local minimizer of min ⟨ones(k,k),L_permutation⟩.

The parameter `L` is a matrix of dimension `n`×`n`,
`k ⩽ n` and `permutation` is a permutation on the rows
and columns of the matrix `L` given as an array of
dimension `n` containing the elements from 1 to `n`.

The local minimizer is stored in the array `permutation`  
The algorithm iteratively exchanges one of the first `k` elements in the
`permutation` with one of the other elements such that in each
step the swap is the best possible at that point until there is
no possible improvement anymore.
"""
function findLocalOptimizer!(permutation,L,k)
    n = size(L,1)
    # initial changeOfSol
    # store in D the change of the solution if
    # we swap (values of) vertex permutation[i] and permutation[k+j]
    D = zeros(k,n-k)
    for i1=1:k
        s = sum(L[permutation[j],permutation[i1]] for j=1:k)
        D[i1,:] .+= L[permutation[i1],permutation[i1]] - 2*s
    end
    for i2=k+1:n
        s = sum(L[permutation[j],permutation[i2]] for j=1:k)
        D[:,i2-k] .+= L[permutation[i2],permutation[i2]] + 2*s 
    end
    D -= 2*L[permutation[1:k],permutation[k+1:n]]

    val = sum(L[permutation[1:k],permutation[1:k]]) # cut value for current permutation
    d = minimum(D) # max improvement

    while d < 0
        # select pair to swap with optimal improvement (randomly)
        poss_swaps = findall(x->x==d, D)
        swap = rand(poss_swaps)
        i1,ci2 = swap[1],swap[2]
        i2 = k + ci2

        impr = D[i1,ci2]
        val += impr
        # update matrix D
        D[i1,:] .-= impr
        D[:,ci2] .-= impr
        for i=vcat(1:i1-1,i1+1:k)
            for j=vcat(1:ci2-1,ci2+1:n-k)
                D[i,j] += 2*(L[permutation[i2],permutation[k+j]] - L[permutation[i1],permutation[k+j]] 
                                + L[permutation[i1],permutation[i]] - L[permutation[i2],permutation[i]])
            end
        end
        # update permutation
        permutation[i1], permutation[i2] = permutation[i2], permutation[i1]
        d = minimum(D)
    end
    return val
end

"""
    function admm_bisect!(bb_node, L, linear_term=missing, const_term=0, max_new_bqpineq=0, upper_bound=Inf; <keyword arguments>)

Compute a lower bound on the generalized `k`-bisection problem with the ADMM
algorithm of de Meijer et al. [[3]](#3).

The generalized `k`-bisection problem means that we want to minimize
the objective `x' * L * x + linear_term' * x + const_term` instead of `x' * L * x`.
In each outer iteration, we add at most `max_new_bqpineq` many new triangle
inequalities.
An `upper_bound` on the objective can be provided. The algorithm stops as soon as
the bound is ≥ this upper bound.
Uses bb_node.estimated_improvement to stop the algorithm if it is not expected to
prune the branch-and-bound node by adding violated triangle inequalities.
If violated triangle inequalities were added, the improvement to the result from
the first outer ADMM loop (DNN bound without triangle inequalities) is updated
in bb_node.estimated_improvement.

# Arguments
- `eps_tols=(1e-3, 5e-3, 1e-3)`: tolerance on when to stop inner ADMM iterations for the
                                 first outer loop (without triangles), the middle part
                                 and the last (final) outer loop.
- `eps_lbimprovement=1e-3`: If the relative improvement of the lower bound is ≤ this parameter,
                            we stop the iteration but do one last round with higher precision.
- `time_limit=Hour(3)`: algorithm stops after given time limit, returns a valid lower bound
- `max_outerloops=100`: the maximum number of iterations adding new trianlge inequalities
- `print_iteration=100`: output after `print_iteration` many inner ADMM iteration steps

# References
<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function admm_bisect!(bb_node, L, linear_term=missing, const_term=0, max_new_bqpineq=0, upper_bound=Inf;
                     eps_tols=(1e-3, 5e-3, 1e-3), eps_lbimprovement=1e-3,
                    time_limit=Hour(3), max_outerloops=100, print_iteration=100)
     
     eps_tol = eps_tols[1]
     eps_tol_middlepart = eps_tols[2]
     eps_tol_lastround = eps_tols[3]
     eps_stagnation = 1e-5
     max_stagnation = 100
     eps_dykstra = 1e-6
     eps_bqpviolation = 1e-3

     n = size(L, 1)
     k = bb_node.k
     if ismissing(linear_term)
          linear_term = spzeros(n)
     end
     if k < n - k
          k = n - k
          Lbar = Symmetric([const_term + 0.5 .* sum(linear_term) -0.125 .* linear_term' 0.125 .* linear_term';
                    -0.125 .* linear_term 0.5 .* (L - 0.5 .* diagm(linear_term)) spzeros(Float64, n, n);
                    0.125 .* linear_term spzeros(Float64, n, n) 0.5 .* (L + 0.5 .* diagm(linear_term))]);
     else
          Lbar = Symmetric([const_term + 0.5 .* sum(linear_term) 0.125 .* linear_term' -0.125 .* linear_term';
                    0.125 .* linear_term 0.5 .* (L + 0.5 .* diagm(linear_term)) spzeros(Float64, n, n);
                    -0.125 .* linear_term spzeros(Float64, n, n) 0.5 .* (L - 0.5 .* diagm(linear_term))]);
     end

     dim_X = 2 * n + 1
     frac = k / n
     V = [1                                    spzeros(Float64, 1, (n - 1));
          frac .* ones(Float64, (n - 1))        sparse(I, (n - 1), (n - 1));
          frac                                -ones(Float64, 1, (n - 1));
          (1 - frac) .* ones(Float64, (n - 1)) -sparse(I, (n - 1), (n - 1));
          (1 - frac)                           ones(Float64, 1, (n - 1))]
     V = Matrix(qr!(V).Q)                      # V'V = I
     Vt = transpose(V)
     R = Symmetric(zeros(n,n))
     X = Symmetric(zeros(dim_X, dim_X)); X[1,1] = 1;
     Z = Symmetric(zeros(dim_X, dim_X))
     sigma = ceil(((2 * n) / k) ^ 2)
     sigma_inv = 1 / sigma
     gamma = 1.608 # gamma = 0.999 for PRSM
     gamma_sigma = gamma * sigma


     # initialization of helper variables
     continue_innerloop = true
     continue_outerloop = true
     first_outerloop = true
     last_outerloop = max_new_bqpineq > 0 ? false : true
     if max_outerloops == 1 # && !last_outerloop
          last_outerloop = true
     end

     counter_stagnation = 0
     counter_innerloops = 0
     counter_outerloops = 0
     counter_iterationstotal = 0

     triangle_ineqs = BQPIneq[]
     clusters_triangle_ineqs = []
     best_lowerbound = -Inf
     old_lowerbound = -Inf
     primal_old = -Inf
     X_old = copy(X)
     best_X = copy(X)
     dnn_bound = -Inf

     start_time = now()
     while continue_outerloop
          println("   primal       dual  err_p_rel   err_d_rel  iteration  time_elapsed")
          counter_outerloops += 1
          while continue_innerloop
               counter_innerloops += 1; counter_iterationstotal += 1;
               XpZ = Symmetric(X + sigma_inv * Z)
               U, ev = projection_PSD_cone2(Symmetric(Vt * (XpZ * V)))
               VU = V * U
               VRVt = Symmetric(VU * diagm(ev) * VU')
               # Z += gamma_sigma * (X - VRVt) for PRSM
               if first_outerloop
                    X = projection_polyhedral(VRVt - sigma_inv * (Lbar + Z), k)
               else
                    X = projection_polyhedral_cyclicdykstra2(VRVt - sigma_inv * (Lbar + Z), clusters_triangle_ineqs, k, eps_dykstra)
               end
               primal_residual = X - VRVt
               Z += gamma_sigma * primal_residual
               primal_obj = dot(Lbar, X)
               dual_obj = dot(Lbar, VRVt)
               if counter_innerloops > 10 && abs(primal_obj - primal_old) < eps_stagnation
                    counter_stagnation += 1
                    if counter_stagnation ≥ max_stagnation
                         continue_innerloop = false
                         println("primal stagnated")
                    end
               end
               dual_residual = Symmetric(Vt * (sigma * (X_old - X) * V))
               #rel_primal_residual = symm_norm(primal_residual) / (sqrt(dim_X) + max(symm_norm(X) , symm_norm(VRVt)))
               #rel_dual_residual = symm_norm(dual_residual) / (sqrt(n) + symm_norm(Symmetric(V' * (Z * V))))
               rel_primal_residual = symm_norm(primal_residual) / (1 + symm_norm(X))
               rel_dual_residual = symm_norm(dual_residual) / (1 + symm_norm(Z))
               X_old = copy(X)
               if rel_dual_residual < eps_tol && rel_primal_residual < eps_tol
                    println("errors small enough")
                    continue_innerloop = false
                    R = U * diagm(ev) * U'
               elseif now() - start_time ≥ time_limit
                    println("timeout")
                    continue_innerloop = false
                    R = U * diagm(ev) * U'
               end
               time_elapsed_s = Dates.value(Millisecond(now() - start_time)) / 10^3 # time elapsed in seconds
               primal_old = primal_obj
               if counter_iterationstotal % print_iteration == 1
                    @printf("%11.5f  %9.5f  %9.7f   %9.7f   %8d   %9.2f s\n", primal_obj, dual_obj,
                              rel_primal_residual, rel_dual_residual, counter_iterationstotal, time_elapsed_s)
               end
          end

          new_lowerbound = compute_valid_lowerbound(Matrix(R), Matrix(Z), triangle_ineqs, V' * (Z * V), Matrix(Lbar), k)[1]
          println("-------------------------------------------------")
          println("valid lower bound: $new_lowerbound")
          println("-------------------------------------------------")
          if counter_outerloops == 1
               eps_tol = eps_tol_middlepart
               dnn_bound = new_lowerbound
               # check whether we expect to be able to prune if we add Δ-inequalities
               if dnn_bound + bb_node.estimated_improvement <= upper_bound - 1
                    println("Do not expect to close integer gap with triangle inequalities")
                    print("estimated improvement is $(bb_node.estimated_improvement), ")
                    println("so $(dnn_bound + bb_node.estimated_improvement) - STOP")
                    last_outerloop = true
               end
               first_outerloop = false
          end
          if new_lowerbound > best_lowerbound
               best_lowerbound = new_lowerbound
               best_X = copy(X)
          end
          if new_lowerbound > (upper_bound - 1) || last_outerloop || now() - start_time ≥ time_limit 
               continue_outerloop = false
          else
               number_newcuts = find_violated_triangles!(triangle_ineqs, X, 2, n+1, max_new_bqpineq, eps_bqpviolation)
               if number_newcuts > 0
                    println("add $number_newcuts new cuts to the problem, in total $(length(triangle_ineqs)).")
                    clusters_triangle_ineqs = cluster_BQPs(triangle_ineqs, 100)
                    @show length(clusters_triangle_ineqs)
               else
                    continue_outerloop = false
               end
               if counter_outerloops == max_outerloops - 1 ||
               abs.(old_lowerbound - new_lowerbound) / old_lowerbound < eps_lbimprovement ||
               number_newcuts < 0.25 * n
                    if number_newcuts < 0.25 * n
                         println("number of new cuts < 0.25n")
                    elseif abs.(old_lowerbound - new_lowerbound) / old_lowerbound < eps_lbimprovement 
                         println("lower bound improvement too slow")
                    end
                    last_outerloop = true
                    eps_tol = eps_tol_lastround
               end
               old_lowerbound = new_lowerbound
               continue_innerloop = true
               counter_stagnation = 0
               counter_innerloops = 0
          end
     end
     if counter_outerloops > 1
          improv = best_lowerbound - dnn_bound
          if improv > 0
               bb_node.estimated_improvement = improv
          end
     end
     return (best_lowerbound, best_X)
end

"""
    symm_norm(A::Symmetric)

Compute the Frobenius norm of a symmetric matrix `A`.

This is added since there is no `symm_norm` for
`Symmetric` matrices, but `dot` is implemented.
"""
function symm_norm(A::Symmetric)
    return sqrt(dot(A,A))
end

"""
    projection_PSD_cone2(M)

Computes the projection of the matrix `M`
onto the cone of positive semidefinite
matrices.

Returns the eigenvectors corresponding to the positive eigenvalues
as columns of the matrix `V` and a vector containing the positive eigenvalues.
"""
function projection_PSD_cone2(M)
    ev, V = eigen(M)
    mp1 = length(ev)
    ind1 = findfirst(>(1e-9), ev)
    V = V[:,ind1:mp1]
    return V, ev[ind1:mp1]
end

"""
    projection_polyhedral(M, k)

Project `M` onto the polyhedral set \$\\mathcal{X}_{BP}\$.

The projection is given in Appendix A of [[3]](#3).

<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function projection_polyhedral(M, k)
    M_proj = Matrix(M)
    n = (size(M,1) - 1) >> 1
    v = (diag(M[2:(n + 1),2:(n + 1)]) - diag(M[(n + 2):end,(n + 2):end])) / 6  +  (M[2:(n + 1),1] - M[(n + 2):end,1]) / 3 .+ 0.5
    v_proj = projection_cappedsimplex(v, k)
    M_proj[1, 1] = 1
    M_proj[1, 2:(n + 1)] = v_proj
    M_proj[1, (n + 2):end] = 1 .- v_proj
    for (i, j) in zip(2:(n + 1), (n + 2):(2 * n + 1))
        M_proj[i,j] = 0
        M_proj[j,j] = M_proj[1,j]
        M_proj[i,i] = M_proj[1,i]
    end
    for j = 3:(n + 1)
        for i = 2:(j - 1)
            if M_proj[i,j] < 0
                M_proj[i,j] = 0
            elseif M_proj[i,j] > 1
                M_proj[i,j] = 1
            end
        end
    end
    # projection box
    for j = (n + 2):(2 * n + 1)
        for i = 2:(j - n - 1)
            if M_proj[i,j] < 0
                M_proj[i,j] = 0
            elseif M_proj[i,j] > 1
                M_proj[i,j] = 1
            end
        end
        for i = (j - n + 1):(j-1)
            if M_proj[i,j] < 0
                M_proj[i,j] = 0
            elseif M_proj[i,j] > 1
                M_proj[i,j] = 1
            end
        end
    end
    return Symmetric(M_proj)
end

"""
    projection_cappedsimplex(y, k)

Return the projection of `y` onto the capped simplex.

The projection onto the capped simplex is
\$\\arg\\min \\lVert x - y \\rVert s.t.: e^\\top x = k, 0 \\leq x \\leq 1\$

Algorithm of [[7]](7).

# Copyright:
Translation of projection.m from
https://github.com/canyilu/Projection-onto-the-capped-simplex

<a id="7">[7]</a>
Weiran Wang and Canyi Lu (2015).
Projection onto the Capped Simplex,
arXiv preprint arXiv:1503.01002.
"""
function projection_cappedsimplex(y, k)
    n = length(y)
    x = zeros(Float64, n)
    if k < 0 || k > n throw(DomainError(k, "argument must be between 0 and length(y)")) end
    if k == 0 return x end
    if k == n return ones(Float64, n) end
    idx = sortperm(y)
    ys = sort(y)

    if k == round(k) # if k is integer
        b = Int(n - k)
        if ys[b+1] - ys[b] >= 1
            x[idx[b+1:end]] .= 1
            return x
        end
    end

    # assume a=0
    s = cumsum(ys)
    ys = vcat(ys, Inf)
    for b = 1:n
        gamma = (k + b - n - s[b]) / b # hypothesized gamma
        if ys[1] + gamma > 0 && ys[b] + gamma < 1 && ys[b+1] + gamma ≥ 1
            x[idx] = vcat(ys[1:b] .+ gamma, ones(n-b))
            return x
        end
    end

    # assume a ≥ 1
    for a = 1:n
        for b = a+1:n
            # hypothesized gamma
            gamma = (k + b - n + s[a] - s[b]) / (b - a)
            if ys[a] + gamma ≤ 0 && ys[a+1] + gamma > 0 && ys[b] + gamma < 1 && ys[b+1] + gamma ≥ 1
                x[idx] = vcat(zeros(a), ys[a+1:b] .+ gamma, ones(n-b))
                return x
            end
        end
    end
    @warn "can not project, try approx method k = $k, n = $n"
    return projection_cappedsimplex_approx(y, k)
    # throw(ErrorException("Error in projection onto capped simplex, did not find root."))
end


"""
    projection_cappedsimplex_approx(y, k, eps=1e-12)

Return the approximate projection of `y` onto the capped simplex.

The projection of `y` onto the capped simplex is
\$\\arg\\min \\lVert x - y \\rVert s.t.: e^\\top x = k, 0 \\leq x \\leq 1\$
with an approximation error of |sum(x)-k| < eps.
The solution is guaranteed to be bounded by 0 and 1.

This is an implementation of Algorithm 1 of [[1]](1).

<a id="1">[1]</a> 
Andersen Ang, Jianzhu Ma, Nianjun Liu, Kun Huang, Yijie Wang (2021).
Fast Projection onto the Capped Simplex with Applications to
Sparse Regression in Bioinformatics,
Advances in Neural Information Processing Systems 34 (NeurIPS 2021) 
"""
function projection_cappedsimplex_approx(y, k, eps=1e-12)
    n = length(y)
    if k == 0
        return zeros(n)
    elseif k == n
        return ones(n)
    elseif k < 0 || k > n
        throw(DomainError(k, "argument must be between 0 and length(y)"))
    end
    gamma = minimum(y) - 0.5
    v = y .- gamma
    w1 = k - sum(v)
    ct = 0
    while abs(w1) > eps
        ct += 1
        w1 = k
        w2 = 0
        for i=1:n
            if v[i] > 0
                if v[i] < 1
                    w1 -= v[i]
                    w2 += 1
                else
                    w1 -= 1
                end
            end
        end
        if w2 == 0
            @warn "k probably too small/large for algorithm, w''(γ) = 0"
            return projection_cappedsimplex(y, k)
        else
            v .+= w1/w2
        end
    end
    #println("number of newton iterations ct = $ct")
    for i=1:n
        if v[i] < 0
            v[i] = 0
        elseif v[i] > 1
            v[i] = 1
        end
    end
    return v
end

"""
    projection_polyhedral_cyclicdykstra2(M, clustersBQP, k, eps)

Project matrix `M` onto the polyhedral set \$\\mathcal{X}_T\$
with clustered cyclic Dykstra's algorithm.

The set of triangles \$T\$ is given in a list of clusters `clustersBQP`.
We stop the iterative projection algorithm as soon as the norm of the
difference of the projections is less than `eps`.

The algorithm is explained in more detail in Section 3.2 of [[3]](#3).

<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function projection_polyhedral_cyclicdykstra2(M, clustersBQP, k, eps)
    X = Symmetric(M)
    X_par = parent(X)
    Nmatrices = Dictionary{Int64, Symmetric}(1:length(clustersBQP), [Symmetric(zeros(Float64,size(M))) for i in 1:length(clustersBQP)])
    N = zeros(size(M))
    diff = 1 + eps
    ct = 0
    while diff > eps
        X_old = copy(X)
        ct += 1
        axpy!(1, parent(N), X_par) # X = X + 1 * N
        N = projection_normalmatrix_polyhedral!(X, k)
        for (ind, bqps) in enumerate(clustersBQP)
            axpy!(1, parent(Nmatrices[ind]), X_par) # X = X + Nmatrices[ind]
            projection_normalmatrix_bqps_parallel!(X, bqps, Nmatrices[ind])
        end
        Xdiff = X_old - X
        diff = sqrt(dot(Xdiff, Xdiff))
    end
    return X
end

"""
    projection_normalmatrix_polyhedral!(M::Symmetric, k)

Project `M` onto the polyhedral set \$\\mathcal{X}_{BP}\$
and return the normal matrix onto the projected matrix.

The projection is given in Appendix A of [[3]](#3).
<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function projection_normalmatrix_polyhedral!(M::Symmetric, k)
    M = parent(M)
    n = (size(M,1) - 1) >> 1
    diagM = diag(M)
    v = (1 / 6) * (diagM[2:(n + 1)] - diagM[(n + 2):end]) + (1 / 3) * (M[1,2:(n + 1)] - M[1,(n + 2):end]) + 0.5 * ones(n)
    v_proj = projection_cappedsimplex(v, k)
    one_min_v_proj = ones(n) - v_proj
    normal_v_proj =  M[1,2:(n + 1)] - v_proj
    normal_v2_proj = M[1,(n + 2):end] - ones(n) + v_proj
    normal_mat = zeros(size(M))
    normal_mat[1,1] = M[1,1] - 1
    M[1,1] = 1;
    normal_mat[1,2:(n + 1)] = normal_v_proj
    M[1,2:(n + 1)] = v_proj
    normal_mat[1,(n + 2):end] = normal_v2_proj
    M[1,(n + 2):end] = one_min_v_proj

    for (ind, i, j) in zip(1:n, 2:(n + 1), (n + 2):(2 * n + 1))
        normal_mat[i,j] = M[i,j]; # normal_mat[j,i] = normal_mat[i,j]
        M[i,j] = 0; #M[j,i] = 0;
        normal_mat[i,i] = M[i,i] - v_proj[ind]
        M[i,i] = v_proj[ind]
        normal_mat[j,j] = M[j,j] - 1 + v_proj[ind]
        M[j,j] = one_min_v_proj[ind]
    end

    # projection box
    for j in 2:(2 * n + 1)
        for i in 2:(j - 1)
            if M[i,j] > 1
                normal_mat[i,j] = M[i,j] - 1
                M[i,j] = 1
            elseif M[i,j] < 0
                normal_mat[i,j] = M[i,j]
                M[i,j] = 0
            end
        end
    end

    return Symmetric(normal_mat)
end

"""
    projection_normalmatrix_bqps_parallel!(M::Symmetric, bqps::Array{BQPIneq,1}, normal_mat)

Project `M` onto H_`bqps` and store in `M` the projection and 
returns the normal matrix `normal_mat`.

This projection is given in Lemma 6 of [[3]](#3).

The triangle inequalities in `bqps` have to be non-overlapping.

It holds `normal_mat` \$ = M - \\mathcal{P}_{H_{bqps}}(M) \$ and
after the call of the function, the projection \$\\mathcal{P}_{H_{bqps}}(M)\$
is stored in `M`.

<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function projection_normalmatrix_bqps_parallel!(M::Symmetric, bqps::Array{BQPIneq,1}, normal_mat)
    M = parent(M)
    normal_mat = parent(normal_mat) #zeros(Float64, size(M))
    fill!(normal_mat, 0)
    for bqp in bqps
        i, j, l = bqp.indices
        lstar = bqp.lstar # lstar = l < n + 2 ? l + n : l - n
        c = (M[l,l] + 2 * (M[1,l] - M[1,lstar]) - M[lstar,lstar] + 3) / 6
        mil = i < l ? M[i,l] : M[l,i];
        mij = i < j ? M[i,j] : M[j,i];
        mjl = j < l ? M[j,l] : M[l,j];
        if mil + mjl <= mij + c            
            normal_mat[l,l] = M[l,l] - c; normal_mat[1,l] = M[1,l] - c;
            normal_mat[lstar,lstar] = M[lstar,lstar] - 1 + c; normal_mat[1, lstar] = M[1,lstar] - 1 + c;
            M[l,l] = c; M[1,l] = c; M[l,1] = c;
            M[lstar,lstar] = 1 - c; M[1,lstar] = 1 - c; M[lstar,1] = 1 - c;
        else
            entry = 0.7 * mil + 0.3 * (-mjl + mij + c)        
            normal_mat[i,l] = mil - entry; normal_mat[l,i] = normal_mat[i,l];
            M[i,l] = entry; M[l,i] = entry;
            entry = entry - mil + mjl        
            normal_mat[j,l] = mjl - entry; normal_mat[l,j] = normal_mat[j,l];
            M[j,l] = entry; M[l,j] = entry;
            entry = -entry + mjl + mij        
            normal_mat[i,j] = mij - entry;
            M[i,j] = entry; M[j,i] = entry;
            entry = 0.1 * (mil + mjl - mij) + 0.9 * c        
            normal_mat[l,l] = M[l,l] - entry; normal_mat[1,l] = M[1,l] - entry;
            M[l,l] = entry; M[1,l] = entry; M[l,1] = entry;
            entry = 1 - entry        
            normal_mat[lstar,lstar] = M[lstar,lstar] - entry; normal_mat[1,lstar] = M[1,lstar] - entry;
            M[lstar,lstar] = entry; M[1,lstar] = entry; M[lstar,1] = entry;
        end
    end
end

"""
    compute_valid_lowerbound(R, Z, bqps, VtZV, Lbar, k)

Compute a valid lower bound and feasible matrix X based on
the output `R`, `Z`, `bqps` of the last ADMM iteration.

The parameter `VtZV` is the matrix V^t⋅Z⋅V which was already
computed in the ADMM algorithm and is provided for performance
reasons.

The valid lower bound computation is done es described in
Section 3.3.1 of [[3]](#3).

<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function compute_valid_lowerbound(R, Z, bqps, VtZV, Lbar, k)
    len = size(Z, 1)
    n = (len - 1) >> 1
    lplb = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    set_silent(lplb)
    set_optimizer_attribute(lplb, "OutputFlag", 0)
    X = @variable(lplb, 0 ≤ X[1:len,1:len] ≤ 1, Symmetric)
    for (i,j) in zip(2:(n + 1), (n + 2):(2 * n + 1))
        @constraint(lplb, X[i,j] == 0)
    end
    @constraint(lplb, X[1,1] == 1)
    @constraint(lplb, sum(X[2:(n + 1),1]) == k)
    @constraint(lplb, sum(X[(n + 2):len,1]) == n - k)
    @constraint(lplb, X[2:(n + 1),1] + X[(n + 2):end,1] .== ones(n))
    @constraint(lplb, [i = 2:len], X[i,1] == X[i,i])
    for bqp in bqps
        i, j, l = bqp.indices
        @constraint(lplb, X[i,l] + X[j,l] <= X[l,l] + X[i,j])
    end
    @objective(lplb, Min, LinearAlgebra.dot(Lbar + Z, X))
    optimize!(lplb)
    opt = objective_value(lplb) - LinearAlgebra.tr(R) * eigmax(VtZV)
    return (opt, value.(X))    
end


"""
    find_violated_triangles!(triangles::Vector{BQPIneq}, X, start_index::Int, end_index::Int, max_newineq, eps=1e-3)

Add to `triangles` all violated triangle inequalities found in
`X[startindex:end_index,start_index:end_index]`.

At most `max_newineq` new triangle inequalities with a violation of at least `eps`
are added to `triangles`. The most violated inequalities found are added.

Returns the number of new triangle inequalties added to `triangles`.
"""
function find_violated_triangles!(triangles::Vector{BQPIneq}, X, start_index::Int, end_index::Int, max_newineq, eps=1e-3)
    if max_newineq == 0 return 0; end
    n = (size(X, 1) - 1) / 2
    @assert start_index > 1 && end_index ≤ size(X, 1)  "Start and end index must be ∈ (1,2n+1]"

    heap = PriorityQueue{BQPIneq, Float64}() # priority queue to help add only the max_newineq most violated ones
    triangles_hash = Set(triangles)          # hash table of all triangles added so far (don't add again)

    for i = start_index:end_index
        for j = i+1:end_index
            for l = j+1:end_index
                xij = X[i,j]; xil = X[i,l]; xjl = X[j,l]; 
                viol = xil + xjl - xij - X[l,l]
                if viol > eps
                    bqp = BQPIneq((i, j, l), l < n + 2 ? l + n : l - n)
                else
                    viol = xij + xjl - xil - X[j,j]
                    if viol > eps
                        bqp = BQPIneq((i, l, j), j < n + 2 ? j + n : j - n)
                    else
                        viol = xij + xil - xjl - X[i,i]
                        if viol > eps
                            bqp = BQPIneq((j, l, i), i < n + 2 ? i + n : i - n)
                        else
                            # no violation
                            @goto after_add_inequality       # @continue only works for not nested ifs
                        end
                    end
                end
                # add triangle inequality to heap
                if bqp ∉ triangles_hash
                    if length(heap) < max_newineq
                        enqueue!(heap, bqp, viol)
                    elseif peek(heap)[2] < viol
                        dequeue!(heap)              # remove triangle inequality with smallest violation
                        enqueue!(heap, bqp, viol)
                    end
                end
                @label after_add_inequality
            end
        end
    end
    append!(triangles, keys(heap))
    return length(heap)
end

"""
    cluster_BQPs(bqps::Array{BQPIneq}, reps=100)

Return list of clusters of non-overlapping BQP inequalities `bqps`.

To find clusters we use the greedy coloring heuristic and the
random_greedy_color heuristic with `reps` repetitions from the
Graph.jl package.
"""
function cluster_BQPs(bqps::Array{BQPIneq}, reps=100)
    G = get_BQPsGraph(bqps)
    coloring = Graphs.Parallel.greedy_color(G)
    coloring2 = Graphs.Parallel.random_greedy_color(G, reps)
    if coloring2.num_colors < coloring.num_colors
        coloring = coloring2
    end
    return [bqps[findall(==(i), coloring.colors)] for i in 1:coloring.num_colors]
end

"""
    get_BQPsGraph(bqps::Array{BQPIneq})

Return the graph representing overlaps of the
BQP-inequalities in `bqps`.

The graph is constructed as describe in the last
paragraph of Section 3.2 in [[3]](#3).

<a id="3">[3]</a> 
de Meijer Frank, Sotirov Renata, Wiegele Angelika, Zhao Shudian (2023).
Partitioning through projections: Strong SDP bounds for large graph partition problems
Comput. Oper. Res., 151.
"""
function get_BQPsGraph(bqps::Array{BQPIneq})
    m = length(bqps)
    G = SimpleGraph(m)
    for i in 1:m
        t1 = bqps[i].indices
        l1l1star = [t1[3], bqps[i].lstar]
        for j in i+1:m
            t2 = bqps[j].indices
            if t2[3] in l1l1star || length(intersect(t1, t2)) > 1
                add_edge!(G, i, j)
            end
        end
    end
    return G
end

"""
    branchingDecision_mostfrac(Xopt)

Return the vertex chosen by the most fractional
branching rule based on `Xopt`.

The vertex chosen by the branching rule
most fractional is the one with value `Xopt[1+i,1+i]`
is closest to 1/2 for 1 ≤ i ≤ n. `Xopt` is the solution of a
relaxation of the original problem with \$\\dim(Xopt) = 2n +1\$.
"""
function branchingDecision_mostfrac(Xopt)
    return argmax(abs.(Xopt[2:Int((size(Xopt,2) - 1) / 2),1] .- 0.5))
end

"""
    branchingDecision_closest2one(Xopt)

Return the vertex chosen by the closest to 1
branching rule based on `Xopt`.

The vertex chosen by the branching rule
closest to 1 is the one with value `Xopt[1+i,1+i]`
is closest to 1 for 1 ≤ i ≤ n. `Xopt` is the solution of a
relaxation of the original problem with \$\\dim(Xopt) = 2n +1\$.
"""
function branchingDecision_closest2one(Xopt)
    return argmax(diag(Xopt)[2:Int((size(Xopt,2) - 1) / 2)])
end

"""
    branch_on_vertex(bbnode::BBNode, vert, set_to)

Return the child BB-node of `bbnode` if vertex `vert`
is set to `set_to`.

The value of `set_to` has to be 0 or 1. 
"""
function branch_on_vertex(bbnode::BBNode, vert, set_to)
    @assert set_to ∈ [0, 1]
    xvec = copy(bbnode.vertices)
    xvec[vert] = set_to
    return BBNode(bbnode.depth + 1, bbnode.lb, xvec, bbnode.k - set_to, bbnode.estimated_improvement)
end

end # module
