using EdgeExpansion
using JSON

"""
    evaluate_dinkelbach(paths; biqbin_path=missing, ncores=4, directory_logfiles=missing,
                                       ignore_alreadyprocessed=true)

Function to run Dinkelbach's algorithm on all directories in `paths`.

Execution is done on all .dat files stored in the directories in `paths`.
If a path `directory_logfiles` is given, then a log file for each instance is stored in
a subdirectory `benchmarkclass/logs_dinkelbach`, where benchmarkclass is the name
of the directory given in `paths`.
If `ignore_alreadyprocessed=true`, then instances with their filename listed in `.ignore-dinkelbach`
in the path given in `paths` are not considered and all instances which were considered within this function
call are added to that file. This allows to interrupt the evaluation and ignore already computed instances.


# Example:
```julia-repl
julia> paths = ["/home/user/data/graphs/grlex/" "/home/user/data/graphs/grevlex/"];
julia> biqbinpath = "/home/user/code/biqbinexpedis/";
julia> dirlogs = "/home/user/data/logs/";
julia> evaluate_dinkelbach(paths, biqbin_path=biqbinpath, ncores=4, directory_logfiles=dirlogs)
```
"""
function evaluate_dinkelbach(paths; biqbin_path=missing, ncores=4, directory_logfiles=missing,
                                    ignore_alreadyprocessed=true)

    for path in paths
        # check path, needed folders and files
        if !(path[end] == '/') path = path * '/' end
        writelogfiles = !ismissing(directory_logfiles)
        dirname = split(path, "/", keepempty=false)[end]
        subdir_logfiles = writelogfiles ? directory_logfiles * "$dirname/logs_dinkelbach/" : missing
        if writelogfiles 
            if !isdir(directory_logfiles) mkdir(directory_logfiles) end
            if !isdir(directory_logfiles * dirname) mkdir(directory_logfiles * dirname) end
            if !isdir(subdir_logfiles) mkdir(subdir_logfiles) end
        end
        toignorefile = ignore_alreadyprocessed ? path * ".ignore-dinkelbach" : missing
        if !isfile(subdir_logfiles * "numerical-problems.txt")
            io = open(subdir_logfiles * "numerical-problems.txt", "w")
        else
            # file for listing instances with numerical problems
            io = open(subdir_logfiles * "numerical-problems.txt", "a")
        end
        
        # get all instances in directory
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
        # sort files by number of vertices
        perm = sortperm_graphFiles(graphFiles, path)

        # compute edge expansion and store result in log file
        for filename in graphFiles[perm]
            # read input
            instancename = String(split(filename, '.')[1])
            filepath = path * filename
            L = EdgeExpansion.RudyGraphIO.laplacian_from_RudyFile(filepath)
            # compute edge expansion
            res_info = try dinkelbach(L, instancename, biqbin_path=biqbin_path, ncores=ncores)
            catch err
                if isa(err, InterruptException)
                    # handle keyboard interrupt
                    close(io)
                    username = ENV["USER"]
                    cmdname = "biqbin"
                    run(`pkill -x $cmdname -u $username`) # stop biqbin as well
                    throw(InterruptException())
                else
                    # handle numerical problems
                    write(io, filename * "\n")
                    username = ENV["USER"]
                    cmdname = "biqbin"
                    try run(`pkill -x $cmdname -u $username`) catch end
                    mv(filepath, path*"processed/"*filename, force=true)
                    continue
                end
            end
            # write result to log file
            if writelogfiles
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
        close(io)
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