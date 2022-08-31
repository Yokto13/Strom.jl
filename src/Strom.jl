module Strom

using Random

include("utils.jl")
include("node.jl")
include("abstracttree.jl")
include("regnode.jl")
include("clsnode.jl")
include("tree.jl")
include("randomforest.jl")
include("boostnode.jl")
include("regboostnode.jl")
include("clsboostnode.jl")
include("boostforest.jl")
include("pca.jl")
include("plot.jl")

export Dato, Data, MSE, MSE_all, invec, onehot, argmax, argmaxx, accuracy, softmax,
       evaluate, setprediction!, RegNode, SMSE, GiniNode, EntropyNode, entropy,
       gini, RegTree, ClsTree, buildtree!, predict, predictall, RandomForest,
       buildforest!, createtrees!, RegBoostNode, ClsBoostNode, BoostForest, 
       gauss_jordan,  read_solution, eig_vals, IPM, eig_vectors, SVD, PCA,
       boostloss, boostloss!, boosttraintest!, boosttraintest, treecontour,
       printtree
export shuffle!

end # module Strom
