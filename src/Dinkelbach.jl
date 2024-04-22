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

using JSON

"""
    dinkelbach_maxcut(L, guess::Rational, graphname)

Compute the edge expansion of a graph given by
its Laplacian matrix `L` applying Dinkelbach's
algorithm.

As a starting `guess` a valid upper
bound on the edge expansion has to be provided.
Returns the edge expansion, the optimal cut and
a dictionary containing a history of the
guesses computed with the Dinkelbach algorithm
as well as the number of B&B nodes and execution
times for each step in the iterative Discrete Newton-
Dinkelbach algorithm.

The underlying optimization problem is solved
by transformation to a max-cut problem and solving
the problem with the max-cut solver of BiqBin [[5]](5).

# References
<a id="5">[5]</a> 
Nicolò Gusmeroli, Timotej Hrga, Borut Lužar, Janez Povh, Melanie Siebenhofer, and Angelika Wiegele (2022).
BiqBin: A Parallel Branch-and-bound Solver for Binary Quadratic Problems with Linear Constraints.
ACM Trans. Math. Softw. 48, 2.
"""
function dinkelbach_maxcut(L, guess::Rational, graphname::String, biqbinpath::String, ncores=4)
    n = size(L,1)
    num = numerator(guess)
    den = denominator(guess)
    xopt = missing
    res_info = Dict{String, Any}()
    res_info["numerators"] = [num]
    res_info["denominators"] = [den]
    res_info["guesses"] = [num/den]
    res_info["bb-time"] = []
    res_info["bb-nodes"] = []
    done = false
    println("Start with guess: $num/$den")
    while !done
        # transform to max cut problem and solve
        filename = graphname*"-n$num-d$den.dat"
        filepath = biqbinpath*"Instances/$filename"
        offset = write_maxcut_inputfile_checkOptimum(L, guess, filepath)
        result_mc = run_maxcut(filepath, biqbinpath, ncores)
        # retrieve solution of max cut
        sol =  offset - result_mc["Solution"]
        println("Solution of bqp was $sol")
        append!(res_info["bb-time"], result_mc["ExecutionTime"])
        append!(res_info["bb-nodes"], result_mc["BabNodes"])
        # do Dinkelbach step
        if sol == 0
            done = true
            println("Guess $guess was the optimum.")
        else
            # retrieve xopt from max cut solution
            indices = result_mc["OneSideOfTheCut"] .- 1
            if 0 in indices
                indices = [i for i in indices if 1 ≤ i ≤ n]
                xopt = zeros(Int64, n)
                xopt[indices] .= 1
            else
                indices = [i for i in indices if 1 ≤ i ≤ n]
                xopt = ones(Int64, n)
                xopt[indices] .= 0
            end
            # new guess based on solution xopt
            den = Int64(sum(xopt))
            if den > n / 2 # sanity check
                @warn "sum(x) > n/2 after biqbin run"
                xopt = abs.(ones(n) - xopt)
                den = sum(xopt)
            end
            num = Int64(xopt' * L * xopt)
            guess = num // den
            println("New guess: $num/$den")
            append!(res_info["numerators"], num)
            append!(res_info["denominators"], den)
            append!(res_info["guesses"], num / den)
        end
    end
    res_info["EdgeExpansion"] = num / den
    return (num // den, xopt, res_info)
end



"""
    write_maxcut_inputfile_checkOptimum(L, guess::Rational, filepath, lower::Int=1, upper::Union{Int,Missing}=missing)

Transform parametric optimization problem to check if `guess` is optimum
to a max-cut instance.

Returns the `offset` and stores the max-cut instance in rudy format
in filepath.
The optimal value of the binary unconstrained optimization problem is equal
to the return value (`offset`) if and only if the edge expansion of the graph
provided by its laplacian matrix `L` is equal to `guess`. 
If the offset minus the optimal value  is <0 (>0) we know
that `guess` is a valid upper (lower) bound for the edge expansion.

It is possible to provide tighter bounds on the smaller set S of the cut
with 1 ≤ `lower` ≤ |S| ≤ `upper` ≤ ⌊n/2⌋.
"""
function write_maxcut_inputfile_checkOptimum(L, guess::Rational, filepath, lower::Int=1,upper::Union{Int,Missing}=missing)
    n = size(L, 1)
    if isequal(upper,missing) upper = Int(floor(n/2)) end
    γn = numerator(guess)
    γd = denominator(guess)
    σ = Int(γn * n + 1)

    ld = Int(ceil(log2(upper - lower + 1)))
    ns = ld - 1

    b1 = lower + 2^ns - (n + 1)/2
    b2 = upper - 2^ns - (n - 1)/2

    offset = - γn*n/2 + σ*(b1^2 + b2^2) - γn*n/2- 3*σ*( n/2*(lower+upper-n) + (2^ns - 1/2)*(upper - 2^ld + 1 - lower) ) + σ*( lower/2*(n - 2^ld + 1) + upper/2*(n+ 2^ld - 1))

    nrvertices = 1 + n + 2*ld
    nredges = Int(n*(n+1)/2 + 2*(n+1)*ld + 2*ld*(ld-1)/2)
    # write rudy input
    io = open(filepath, "w")
    write(io, "$nrvertices $nredges\n")
    # row zero of adjacency, 1 ≤ j ≤ n
    entry = Int(-γn - 2*σ*(b1+b2))
    for j=2:n+1
        write(io, "1 $j $entry\n")
    end
    factor1 = 2*σ*b1 # first binary slack part
    factor2 = -2*σ*b2 # second binary slack part
    # row zero, j > n
    ind = n+2
    for k = 0:ns
        # i = 0, j in binary slack part
        twopowk = 2^k
        write(io, "1 $ind $(Int(factor1*twopowk)) \n1 $(ind + ld) $(Int(factor2*twopowk)) \n")
        # 1 ≤ i ≤ n, j in binary slack part
        entry = Int(σ*twopowk)
        for i = 2:n+1
            write(io, "$i $ind $(-entry) \n$i $(ind+ld) $entry\n")
        end
        ind += 1
    end
 
    # i and j in first binary slack part and
    # i and j in second binary slack part
    ind = n + 3
    for k = 1:ns
        indrow = n + 2
        for l = 0:k-1
            entry = Int(σ*(2^(k+l)))
            write(io, "$indrow $ind $entry\n$(indrow+ld) $(ind+ld) $entry \n")
            indrow += 1
        end
        ind += 1
    end

    # 1 ≤ i < j ≤ n
    sigma2 = 2*σ
    for i = 1:n
        for j = i+1:n
            write(io, "$(i+1) $(j+1) $(γd*L[i,j] + sigma2) \n")
        end
    end
    close(io)
    return offset
end

"""
    run_maxcut(filepath, biqbinpath)

Solve the max cut problem with rudy input file stored in `filepath`.

Returns dictionary with the "ExecutionMD" part of the JSON output file of
BiqBin's max cut solver [[5]](5).
The path to the BiqBin installation has to be provided, for example
`biqbinpath` = "home/user/src/biqbin-expedis/"

# References
<a id="5">[5]</a> 
Nicolò Gusmeroli, Timotej Hrga, Borut Lužar, Janez Povh, Melanie Siebenhofer, and Angelika Wiegele (2022).
BiqBin: A Parallel Branch-and-bound Solver for Binary Quadratic Problems with Linear Constraints.
ACM Trans. Math. Softw. 48, 2.
"""
function run_maxcut(filepath, biqbinpath, ncores=4)
    mpirunexe = "/usr/bin/mpirun"
    # run maxcut code
    result = read(`$mpirunexe -n $ncores $(biqbinpath)biqbin $filepath $(biqbinpath)params Julia 10 100`, String)
    # remove temporary solution files in directory
    foreach(rm, filter(startswith("$(filepath)_Julia_"), readdir("$(filepath[1:findlast("/", filepath)[1]])",join=true)))
    return JSON.parse(result)["ExecutionMD"]
end