using EdgeExpansion

"""
    evaluate_splitandbound(paths; biqbin=true, biqbin_path=missing, ncores=4)

Function to run split and bound code on all directories in `paths`.

Execution is done on all .dat files stored in the directories in `paths`.
After computation the input file is moved to the subdirectory /processed.
In /logs a logfile in JSON format is stored.

# Example:
```julia-repl
julia> paths = ["/home/user/data/graphs/grlex/" "/home/user/data/graphs/grevlex/"];
julia> biqbinpath = "/home/user/code/biqbinexpedis/";
julia> evaluate_splitandbound(paths, biqbin=true, biqbin_path=biqbinpath, ncores=4);
```
"""
function evaluate_splitandbound(paths; biqbin=true, biqbin_path=missing, ncores=4)
    for path in paths
        if !(path[end] == '/') path = path * '/' end
        if !isdir(path*"logs/") mkdir(path*"logs/") end
        if !isdir(path*"processed/") mkdir(path*"processed/") end
        graphFiles = filter(x->endswith(x, ".dat"), readdir(path,sort=false))
        perm = sortperm_graphFiles(graphFiles, path)
        for filename in graphFiles[perm]
            # input
            filepath = path * filename
            instancename = String(split(filename, '.')[1])
            L = EdgeExpansion.RudyGraphIO.laplacian_from_RudyFile(filepath)

            # split and bound
            solutions_dict = split_and_bound(L, instancename, biqbin=biqbin, biqbin_path=biqbin_path, ncores=ncores) 

            # write logfile
            io = open(path * "logs/" * instancename* ".json", "w")
            JSON.print(io, solutions_dict, 4)
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
