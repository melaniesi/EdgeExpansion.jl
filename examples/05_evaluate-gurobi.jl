using EdgeExpansion

"""

Function to compute the edge expansion with Gurobi on all
instances in the directories listed in `paths`.

The computation is performed on all .dat files. After computation
the input file is moved to the subdirectory /processed
In /logs_gurobi a logfile with objective bound, relative gap
and computation time is stored.

# Example:
```julia-repl
julia> paths = ["./graphs/grlex/", "./graphs/grevlex/"];
julia> evaluate_gurobi(paths)
```
"""
function evaluate_gurobi(paths, uselowerbound=true; nthreads=4)
    for path in paths
        if !(path[end] == '/') path = path * '/' end
        if !isdir(path*"logs_gurobi/") mkdir(path*"logs_gurobi/") end
        if !isdir(path*"processed/") mkdir(path*"processed/") end
        graphFiles = filter(x->endswith(x, ".dat"), readdir(path,sort=false))
        perm = sortperm_graphFiles(graphFiles, path)
        for filename in graphFiles[perm]
            # input
            filepath = path * filename
            instancename = String(split(filename, '.')[1])
            L = EdgeExpansion.RudyGraphIO.laplacian_from_RudyFile(filepath)

            # solve with gurobi
            val, gap, t  = uselowerbound ? gurobi_solve_exact(L, lb=0, timelimit=10800.0, nrthreads=nthreads)  :
                                           gurobi_solve_exact(L, timelimit=10800.0, nrthreads=nthreads) 

            # write logfile
            io = open(path * "logs_gurobi/" * instancename* ".csv", "w")
            write(io, "instance;n;objectivebound;relativegap;solvetime\n")
            write(io, "$instancename;$(size(L,1));$val;$gap;$t\n")
            close(io)

            # move file to processed folder
            mv(filepath, path*"processed/"*filename, force=true)
        end
    end
end

"""
    sortperm_graphFiles(graphFiles, path_dir)

Return a permutation vector sorted by the number of vertices
of the graphs listed in `graphFiles` and stored in `path_dir`.

# Arguments:
- `graphFiles`: Array of graph file names to be sorted by the
                number of vertices of the corresponding graph
                in ascending order.
- `path_dir::String:` Path to the directory the files in
                      `graphFiles` are stored.
"""
function sortperm_graphFiles(graphFiles, path_dir)
    nrVertices = similar(graphFiles, Int64)
    for (i, filename) in enumerate(graphFiles)
        io = open(path_dir * filename)
        nrVertices[i], _ = parse.(Int64, split(readline(io)," "))
        close(io)
    end
    return sortperm(nrVertices)
end

gurobi_solve_exact(EdgeExpansion.PolytopeGraphGenerator.grevlex(3), lb = 0);
