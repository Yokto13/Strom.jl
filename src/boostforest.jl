using Random

include("node.jl")
include("utils.jl")
include("tree.jl")
include("regboostnode.jl")
include("clsboostnode.jl")

mutable struct BoostForest
    data
    trees::Union{Vector, Vector{Vector}}
    treecnt::Integer
    node::BoostNode 
    minnode::Integer
    maxdepth::Integer
    bagging::Bool
    ftrsubset::Function
    datasetsubset::Function
    G::Vector
    H::Vector
    logits::Matrix
    probs::Matrix
    treestrained::Integer
    α::Number
    λ::Number
end

getindex(f::BoostForest, i::Integer) = (f.trees[i])
getindex(f::BoostForest, i::Integer, j::Integer) = (f.trees[i, j])

BoostForest(data, treecnt, node) = BoostForest(data, 
                                               [], 
                                               treecnt,      
                                               node, 
                                               1, 
                                               1000, 
                                               true,       
                                               x -> x,
                                               x -> x,
                                               [], 
                                               [], 
                                               reshape([],0,1),
                                               reshape([],0,1),
                                               0,
                                               0.1,
                                               1
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
             [], 
             [], 
             reshape([],0,1),
             reshape([],0,1),
             0,
             0.1,
             1
            )

function updateG!(forest::BoostForest, c=nothing)
    preds = predictall(forest.data, forest)
    data = forest.data
    if forest.node isa RegBoostNode
        forest.G = [preds[i] - forest.data[i].y for i=1:length(forest.data)]
    end
    if forest.node isa ClsBoostNode
        targets = [d.y for d in forest.data]
        forest.G = forest.probs[:, c] - (targets .== c)
    end
end

function updateH!(forest::BoostForest, c=nothing)
    if forest.node isa RegBoostNode
        forest.H = ones(length(forest.data))
    end
    if forest.node isa ClsBoostNode
        forest.probs[:, c] = softmax(forest.logits[:, c])
        forest.H = forest.probs[:, c] .* (1 .- forest.probs[:, c])
    end
end

"""
    predict(datapoint, forest)

Get prediction for the given `datapoint`.
"""
function predict(datapoint, forest::BoostForest)
    pred = 0 
    for i=1:forest.treestrained
        p = predict(datapoint, forest.trees[i])
        pred = pred .+ forest.α * p
    end
    return pred
end

function predict(datapoint, timestamp::Vector)
    pred = zeros(length(timestamp))
    for c=1:length(timestamp)
        pred[c] = predict(datapoint, timestamp[c])
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
    if inittrees
        createtrees!(forest)
    end
    @assert length(forest.trees) != 0
    buildtrees!(forest, forest.node)
end

function buildtrees!(forest::BoostForest, node::RegBoostNode)
    updateH!(forest)
    updateG!(forest)
    for tree=forest.trees
        tree.G = forest.G
        tree.H = forest.H
        buildtree!(tree)
        forest.treestrained += 1
        updateH!(forest)
        updateG!(forest)
    end
end

function initpreds!(forest::BoostForest)
    forest.probs = zeros(length(forest.data), forest.data.classcnt)
    forest.logits = zeros(length(forest.data), forest.data.classcnt)
end

function buildtrees!(forest::BoostForest, node::ClsBoostNode)
    initpreds!(forest)
    for timestamp=1:forest.treecnt
        for c=1:forest.data.classcnt
            updateH!(forest, c)
            updateG!(forest, c)
            forest[timestamp][c].G = forest.G
            forest[timestamp][c].H = forest.H
            buildtree!(forest[timestamp][c])
            updateH!(forest, c)
            updateG!(forest, c)
            preds = predictall(forest.data, forest[timestamp][c])
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
    for i=1:forest.treecnt
        push!(forest.trees, createtimestamp(forest, forest.node))
    end
end

function createtimestamp(forest::BoostForest, node::RegBoostNode)
    data = forest.data
    treedata = Data(deepcopy(data.data), data.classcnt)
    treedata.data = treedata
    t = Tree(treedata, deepcopy(node), forest.minnode,
             forest.maxdepth, forest.λ)
    return t
end

function createtimestamp(forest::BoostForest, node::ClsBoostNode)
    timestamp = []
    data = forest.data
    for i=1:data.classcnt
        treedata = Data(deepcopy(data.data), data.classcnt)
        treedata.data = treedata
        t = Tree(treedata, deepcopy(node), forest.minnode,
             forest.maxdepth)
        t.λ = forest.λ
        push!(timestamp, t)
    end
    return timestamp
end
