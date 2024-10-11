using EdgeExpansion
using JSON

"""
    evaluate_splitandbound(paths; biqbin=true, biqbin_path=missing, ncores=4, directory_logfiles=missing,
                                       ignore_alreadyprocessed=true)

Function to run split and bound code on all directories in `paths`.

Execution is done on all .dat files stored in the directories in `paths`.
If a path `directory_logfiles` is given, then a log file for each instance is stored in
a subdirectory `benchmarkclass/logs_splitandbound`, where benchmarkclass is the name
of the directory given in `paths`.
If `ignore_alreadyprocessed=true`, then instances with their filename listed in `.ignore-splitandbound`
in the path given in `paths` are not considered and all instances which were considered within this function
call are added to that file. This allows to interrupt the evaluation and ignore already computed instances.

# Example:
```julia-repl
julia> paths = ["/home/user/data/graphs/grlex/" "/home/user/data/graphs/grevlex/"];
julia> biqbinpath = "/home/user/code/biqbinexpedis/";
julia> dirlogs = "/home/user/data/logs/";
julia> evaluate_splitandbound(paths, biqbin=true, biqbin_path=biqbinpath, ncores=4, directory_logfiles=dirlogs);
```
"""
function evaluate_splitandbound(paths; biqbin=true, biqbin_path=missing, ncores=4, directory_logfiles=missing,
                                       ignore_alreadyprocessed=true)
    for path in paths
        if !(path[end] == '/') path = path * '/' end
        writelogfiles = !ismissing(directory_logfiles)
        dirname = split(path, "/", keepempty=false)[end]
        subdir_logfiles = writelogfiles ? directory_logfiles * "$dirname/logs_splitandbound/" : missing
        if writelogfiles 
            if !isdir(directory_logfiles) mkdir(directory_logfiles) end
            if !isdir(directory_logfiles * dirname) mkdir(directory_logfiles * dirname) end
            if !isdir(subdir_logfiles) mkdir(subdir_logfiles) end
        end
        toignorefile = ignore_alreadyprocessed ? path * ".ignore-splitandbound" : missing

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

            # split and bound
            solutions_dict = split_and_bound(L, instancename, biqbin=biqbin, biqbin_path=biqbin_path, ncores=ncores) 

            # write logfile
            if writelogfiles
                io = open(subdir_logfiles * instancename* ".json", "w")
                JSON.print(io, solutions_dict, 4)
                close(io)
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
