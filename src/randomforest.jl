using Random

mutable struct RandomForest 
    data
    trees::Vector
    treecnt::Integer
    node::Node
    minnode::Integer
    maxdepth::Integer
    bagging::Bool
    ftrsubset::Function
    datasetsubset::Function
end

RandomForest(data, treecnt, node) = RandomForest(data, [], treecnt, 
                                                     node, 1, 1000, true, 
                                                     x -> Integer(ceil(sqrt(x))),
                                                     x -> Integer(ceil(sqrt(x))))

RandomForest(data, treecnt, node, minnode=1, maxdepth=10,) = 
RandomForest(data, 
             [], 
             treecnt, 
             node, 
             minnode, 
             maxdepth, 
             true,
             x -> Integer(ceil(sqrt(x))),
             x -> Integer(ceil(sqrt(x)))
           )

"""
    predict(datapoint, forest)

Get prediction for the given `datapoint`.
"""
function predict(datapoint, forest::RandomForest)
    pred = nothing
    for tree=forest.trees
        if isnothing(pred)
            pred = predict(datapoint, tree)
        else
            pred += predict(datapoint, tree)
        end
    end
    return pred / length(forest.trees)
end

"""
    predictall(datapoints, forest)

Get predictions for all `datapoints`, return them as a vector.
"""
function predictall(datapoints, forest::RandomForest)
    predictions = []
    for dp=datapoints
        push!(predictions, predict(dp, forest))
    end
    return predictions
end

"""
    buildforest!(forest[,inittrees])

Build forest by repeatedly building each tree.

By default `inittrees` is true and trees are instatiated by the function.
You can also instantied them by yourself then set `inittrees` false 
and `forest.trees` and `forest.treecnt` accordingly.
"""
function buildforest!(forest::RandomForest, inittrees::Bool=true)
    if inittrees
        createtrees!(forest)
    end
    for tree=forest.trees
        buildtree!(tree)
    end
end

"""
    createtrees!(forest)

Init `forest.treecnt` trees and placed them to `forest.trees`.

Uses params specified in the `forest`.
"""
function createtrees!(forest::RandomForest)
    forest.trees = []
    datasetsize = forest.datasetsubset(length(forest.data))
    for i=1:forest.treecnt
        treedata = Data(deepcopy(forest.data.data), forest.data.classcnt)
        shuffle!(treedata.data)
        treedata.data = treedata[1:datasetsize]
        t = Tree(treedata, deepcopy(forest.node), forest.minnode,
                 forest.maxdepth)
        push!(forest.trees, t)
    end
end
