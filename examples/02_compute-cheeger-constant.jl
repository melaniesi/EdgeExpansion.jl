using Revise
using EdgeExpansion

# set path to biqbin
biqbin_path="/home/users/mesiebenhofe/Dokumente/Mathematik/01_Dissertation/02_Code/biqbin-expedis/"

L = EdgeExpansion.PolytopeGraphGenerator.grevlex(6)
res = split_and_bound(L, biqbin_path=biqbin_path, ncores=4) # split and bound with biqbin
res = split_and_bound(L, biqbin=false) # split and bound with bisection branch-and-bound

instance_path = "./rand01-7-32-0.dat"
L = EdgeExpansion.RudyGraphIO.laplacian_from_RudyFile(instance_path)
res = split_and_bound(L, biqbin_path=biqbin_path, ncores=4)