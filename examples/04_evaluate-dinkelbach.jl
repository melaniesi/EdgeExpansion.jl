using EdgeExpansion

"""
    evaluate_dinkelbach(paths; biqbin_path=missing, ncores=4)

Function to run Dinkelbach's algorithm on all directories in `paths`.

Execution is done on all .dat files stored in the directories in `paths`.
After computation the input file is moved to the subdirectory /processed.
In /logs a logfile in JSON format is stored.

# Example:
```julia-repl
julia> paths = ["/home/user/data/graphs/grlex/" "/home/user/data/graphs/grevlex/"];
julia> biqbinpath = "/home/user/code/biqbinexpedis/";
julia> evaluate_dinkelbach(paths biqbin_path=biqbinpath, ncores=4)
```
"""
function evaluate_dinkelbach(paths; biqbin_path=missing, ncores=4)

    for path in paths
        # check needed folders and files
        if !isdir(path*"/logs_dinkelbach/") mkdir(path*"/logs_dinkelbach/") end
        if !isfile(path*"/logs_dinkelbach/numerical-problems.txt")
            io = open(path*"/logs_dinkelbach/numerical-problems.txt", "w")
        else
            # file for listing instances with numerical problems
            io = open(path*"/logs_dinkelbach/numerical-problems.txt", "a")
        end
        if !isdir(path*"/processed/") mkdir(path*"/processed/") end
        # sort files by number of vertices
        graphFiles = filter(x->endswith(x, ".dat"), readdir(path,sort=false))
        perm = sortperm_graphFiles(graphFiles, path)
        # compute edge expansion and store result in log file + move file to processed/
        for filename in graphFiles[perm]
            # read input
            instancename = split(filename, '.')[1]
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
                    mv(filepath, path*"/processed/"*filename, force=true)
                    continue
                end
            end
            # write result to log file
            io_res = open(path*"logs_dinkelbach/"*instancename*".json", "w")
            JSON.print(io_res, res_info, 4)
            close(io_res)
            # move file to /processed/
            mv(filepath, path*"/processed/"*filename, force=true)
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