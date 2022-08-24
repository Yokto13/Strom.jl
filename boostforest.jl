using Random

include("node.jl")
include("tree.jl")

mutable struct BoostForest
    data
    trees::Vector
    treecnt::Integer
    node 
    minnode::Integer
    maxdepth::Integer
    bagging::Bool
    ftrsubset::Function
    datasetsubset::Function
    G::Vector
    H::Vector
    logits::Vector
    probs::Vector{Vector}
    treestrained::Integer
end

getindex(f::BoostedForest, inds) = (f.trees[inds])

BoostForest(data, treecnt, node) = BoostForest(data, [], treecnt, 
                                                     node, 1, 1000, true, 
                                                     x -> x,
                                                     x -> x,
                                                     [], [], 0
                                                    )

BoostForest(data, treecnt, node, minnode=1, maxdepth=10,) = 
BoostForest(data, 
             [], 
             treecnt, 
             node, 
             minnode, 
             maxdepth, 
             true,
             x -> x,
             x -> x,
             [], [], 0
           )

function updateG!(forest::BoostForest)
    preds = predictall(forest.data, forest)
    data = forest.data
    forest.G = [preds[i] - forest.data[i].y for i=1:length(forest.data)]
end

function updateH!(forest::BoostForest, c=nothing)
    if typeof(forest.node) === RegBoostNode
        forest.H = ones(length(forest.data))
    end
    if typeof(forest.node) === ClsBoostNode
        forest.H = forest.probs[:, c] .* (1 .- forest.probs[:, c])
    end
end

"""
    predict(datapoint, forest)

Get prediction for the given `datapoint`.
"""
function predict(datapoint, forest::BoostForest)
    pred = 0.0
    if forest.treestrained == 0
        return pred
    end
    for i=1:forest.treestrained
        p = predict(datapoint, forest.trees[i])
        # println(p)
        pred += p * 0.5
    end
    return pred
end

"""
    predictall(datapoints, forest)

Get predictions for all `datapoints`, return them as a vector.
"""
function predictall(datapoints, forest::BoostForest)
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
function buildforest!(forest::BoostForest, inittrees::Bool=true)
    updateH!(forest)
    updateG!(forest)
    if inittrees
        createtrees!(forest)
    end
    @assert length(forest.trees) != 0
    buildtrees!(forest, forest.node)
end

function buildtrees!(forest::BoostForest, node::RegBoostNode)
    for tree=forest.trees
        tree.G = forest.G
        tree.H = forest.H
        buildtree!(tree)
        forest.treestrained += 1
        updateH!(forest)
        updateG!(forest)
    end
end

function buildtrees!(forest::BoostForest, node::ClsBoostNode)
    forest.logits = zeros(length(forest.data), forest.data.classcnt)
    for timestamp=1:forest.treecnt
        for c=1:forest.data.classcnt
            forest[timestamp, c].G = forest.G
            forest[timestamp, c].H = forest.H
            buildtree!(forest[timestamp, c])
            updateH!(forest, c)
            updateG!(forest)
            preds = predictall(forest[timestamp, c], data)
            preds = sum(preds, 1)
            forest.logits[:, c] += preds
        end
        forest.treestrained += 1
    end
end

"""
    createtrees!(forest)

Init `forest.treecnt` trees and placed them to `forest.trees`.

Uses params specified in the `forest`.
"""
function createtrees!(forest::BoostForest)
    forest.trees = []
    # datasetsize = forest.datasetsubset(length(forest.data))
    for i=1:forest.treecnt
        # This is sooooo inefficient
        # TODO solve deep copy thing
        # TODO some profiling
        treedata = Data(deepcopy(forest.data.data), forest.data.classcnt)
        # shuffle!(treedata.data)
        treedata.data = treedata
        t = Tree(treedata, deepcopy(forest.node), forest.minnode,
                 forest.maxdepth)
        push!(forest.trees, t)
    end
end
