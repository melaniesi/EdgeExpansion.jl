using EdgeExpansion
using JSON

"""

Function to compute the edge expansion with Gurobi on all
instances in the directories listed in `paths`.

The computation is performed on all .dat files. 
If a path `directory_logfiles` is given, then a log file for each instance is stored in
a subdirectory `benchmarkclass/logs_gurobi`, where benchmarkclass is the name
of the directory given in `paths`.
If `ignore_alreadyprocessed=true`, then instances with their filename listed in `.ignore-gurobi`
in the path given in `paths` are not considered and all instances which were considered within this function
call are added to that file. This allows to interrupt the evaluation and ignore already computed instances.

# Example:
```julia-repl
julia> paths = ["./graphs/grlex/", "./graphs/grevlex/"];
julia> dirlogs = "./logs/";
julia> evaluate_gurobi(paths, directory_logfiles=dirlogs)
```
"""
function evaluate_gurobi(paths, uselowerbound=true; nthreads=4, directory_logfiles=missing, ignore_alreadyprocessed=true)
    for path in paths
        if !(path[end] == '/') path = path * '/' end
        writelogfiles = !ismissing(directory_logfiles)
        dirname = split(path, "/", keepempty=false)[end]
        subdir_logfiles = writelogfiles ? directory_logfiles * "$dirname/logs_gurobi/" : missing
        if writelogfiles 
            if !isdir(directory_logfiles) mkdir(directory_logfiles) end
            if !isdir(directory_logfiles * dirname) mkdir(directory_logfiles * dirname) end
            if !isdir(subdir_logfiles) mkdir(subdir_logfiles) end
        end
        toignorefile = ignore_alreadyprocessed ? path * ".ignore-gurobi" : missing

        graphFiles = filter(x->endswith(x, ".dat"), readdir(path,sort=false))

        # remove instances to ignore
        if ignore_alreadyprocessed && isfile(toignorefile)
            toignore = []
            io = open(toignorefile, "r")
            while !eof(io)
                push!(toignore, readline(io))
            end
            close(io)
            deleteat!(graphFiles, findall(in(toignore), graphFiles))
        end

        perm = sortperm_graphFiles(graphFiles, path)
        for filename in graphFiles[perm]
            # input
            filepath = path * filename
            instancename = String(split(filename, '.')[1])
            L = EdgeExpansion.RudyGraphIO.laplacian_from_RudyFile(filepath)

            # solve with gurobi
            val, gap, t  = uselowerbound ? gurobi_solve_exact(L, lb=0, timelimit=10800.0, nrthreads=nthreads)  :
                                           gurobi_solve_exact(L, timelimit=10800.0, nrthreads=nthreads) 

             # write result to log file
            if writelogfiles
                res_info = Dict("objectivebound" => val, "relativegap" => gap, "time" => t)
                io_res = open(subdir_logfiles*instancename*".json", "w")
                JSON.print(io_res, res_info, 4)
                close(io_res)
            end

            # mark instance as processed
            if ignore_alreadyprocessed
                io = open(toignorefile, "a")
                write(io, "\n" * filename)
                close(io)
            end
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
