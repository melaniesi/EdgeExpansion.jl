using EdgeExpansion
using EdgeExpansion.PolytopeGraphGenerator
using EdgeExpansion.RudyGraphIO

instances_directory = "/home/users/mesiebenhofe/Dokumente/Mathematik/01_Dissertation/04_Data/graphs/"

L = grevlex(7)
filepath = instances_directory * "grevlex/grevlex-7.dat"
laplacian_to_RudyFile(L, filepath)

L = grlex(7)
filepath = instances_directory * "grlex/grlex-7.dat"
laplacian_to_RudyFile(L, filepath)

L = laplacian_rand01polytope(75, 9)
filepath = instances_directory * "rand01-polytopes/rand01-9-75-0.dat"
laplacian_to_RudyFile(L, filepath)