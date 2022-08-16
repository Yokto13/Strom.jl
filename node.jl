using Random 

rng_seed = 42
Random.seed!(rng_seed)

include("utils.jl")

abstract type Node end
abstract type ClsNode <: Node end

mutable struct RegNode <: Node
    isleaf::Bool
    pred::Numeric64
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
end

RegNode(v::Vector{Int64}) = RegNode(false, -1, v, -1, -1, nothing, nothing, 1)
RegNode() = RegNode(false, -1, [], -1, -1, nothing, nothing, 1)

mutable struct GiniNode <: ClsNode
    isleaf::Bool
    pred::Vector
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
end

GiniNode(v::Vector{Int64}) = GiniNode(false, [], v, -1, -1, nothing, nothing, 1)
GiniNode() = GiniNode(false, [], [], -1, -1, nothing, nothing, 1)

mutable struct EntropyNode <: ClsNode
    isleaf::Bool
    pred::Vector
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
end

mutable struct Tree
    data
    root::Node
    minnode
    maxdepth
end

RegTree(data) = Tree(data, RegNode(), 1)
RegTree(data, minnode, maxdepth) = Tree(data, RegNode(), minnode, maxdepth)

ClsTree(data) = Tree(data, GiniNode(), 1, nothing)
ClsTree(data, minnode, maxdepth) = Tree(data, GiniNode(), minnode, maxdepth)

""" 
    calcprediction(n, data)

Calculates prediction for `n` on `n.datainds`.

This function should be used *only* in setprediction!
and should **not** be exposed to the outside.
For predictions to be sensible, 
followup actions ought to be done in setprediction!.
"""
function calcprediction(n::RegNode, data)
    pred::Float64 = 0.0
    sz = length(n.datainds)
    for j = 1:sz
        i = n.datainds[j]
        pred += data[i].y
    end
    return pred 
end

function calcprediction(n::ClsNode, data)
    pred = zeros(data.classcnt)
    sz = length(n.datainds)
    for j = 1:sz
        i = n.datainds[j]
        pred[data[i].y] += 1
    end
    return pred
end

"""
    setprediction!(n, data)

Calculates prediction and sets it to `n.pred`.
"""
function setprediction!(n, data)
    pred = calcprediction(n, data)
    pred /= length(n.datainds)
    n.pred = pred
end

"""
    stopdividing(n[, cond])

Determine if node dividing should halt.

If `cond` is unspecified, apply x -> length(x.datainds) < 2.
"""
function stopdividing(n::Node,
        cond::Union{Function, Bool}= x -> length(x.datainds) < 2)
    if cond(n)
        return true
    end
    return false
end

"""
    stopdividing(n, tree)
"""
function stopdividing(n::Node, tree::Tree)
    cond = get_stopcondition(tree)
    cond = x -> length(x.datainds) < tree.minnode
    return stopdividing(n, cond)
end

"""
    get_stopcondition(tree)

Creates partial function from `tree` which can determine node split.
"""
function get_stopcondition(tree::Tree)
    return x -> ((!isnothing(tree.maxdepth) && x.depth >= tree.maxdepth) 
                 || (!isnothing(tree.minnode) && length(x.datainds) < tree.minnode)
                )
end

"""
    evaluate(n, data)

Evaluate node.
"""
function evaluate(n::RegNode, data)
    return SMSE(n, data)
end
function evaluate(n::GiniNode, data)
    return gini(n, data)
end
function evaluate(n::EntropyNode, data)
    return entropy(n, data)
end
"""
    evaluate(left, right, data)

Calculate error for left and right and get its sum.

Used to score how good a certain node split is.
"""
function evaluate(left::Node, right::Node, data)
    return evaluate(left, data) + evaluate(right, data)
end

"""
    SMSE(n, data)

Sum of mean squared errors belonging to `n.datainds`,
"""
function SMSE(n::Node, data)
    sz = length(n.datainds)
    res::Float64 = 0.0
    for j = 1:sz
        i = n.datainds[j]
        res += (n.pred - data[i].y)^2
    end
    #if sz != 0
    #    res /= sz
    #end
    return res
end

"""
    entropy(n, data)

Get the entropy criterion for prediction in the given node `n`.

One possible criterion for classification.
Adds small epsilon to all values in `n.pred` so 0 in log is avoided.
"""
function entropy(n::EntropyNode, data)
    eps = 1e-9
    return - length(n.datainds) * sum(n.pred .* log2.(n.pred .+ eps))
end

"""
    gini(n, data)

Get the gini criterion for prediction in the given node `n`.

One possible criterion for classification.
"""
function gini(n::GiniNode, data)
    return length(n.datainds) * sum(n.pred .* (1 .- n.pred))
end

"""
    splitnode(ftr, val, node, data)

Get child-nodes containing `datainds` split on feature `ftr` and value `val`.

All points < `val` are placed to the `left`, the rest to the `right`. 
"""
function splitnode(ftr, val, n::Node, data)
    left, right = typeof(n)(), typeof(n)()
    left.depth = right.depth = n.depth + 1
    for j = 1:length(n.datainds)
        if data[n.datainds[j]][ftr] < val
            push!(left.datainds, n.datainds[j])
        else
            push!(right.datainds, n.datainds[j])
        end
    end
    return (left, right)
end

"""
    evaluatesplit(ftr, val, n, data)
"""
function evaluatesplit(ftr::Int64, val::Float64, n::Node, data)
    left, right = splitnode(ftr, val, n, data)
    setprediction!(left, data)
    setprediction!(right, data)
    return evaluate(left, right, data)
end

"""
    getextremas(v)

Find min and max value in `v`.
"""
function getextremas(v::Vector)
    mi::Float64 = Inf
    ma::Float64 = -Inf
    for x = v
        mi = min(mi, x)
        ma = max(ma, x)
    end
    return (mi, ma)
end

"""
    getextremas(feature, n, data)

Find min and max of `ftr` in `n.datainds`.
"""
function getextremas(ftr::Int64, n::Node, data)
    mi::Float64 = Inf
    ma::Float64 = -Inf
    for j = 1:length(n.datainds)
        ex_i = n.datainds[j]
        mi = min(mi, data[ex_i][ftr])
        ma = max(ma, data[ex_i][ftr])
    end
    @assert(mi != Inf)
    @assert(ma != -Inf)
    return (mi, ma)
end

"""
    gensplits(mi, ma, cnt)

Generate `cnt` float values uniformly between `mi` and `ma`.
"""
function uniform(mi, ma, cnt)
    out = rand(cnt)
    @assert(ma >= mi)
    diff = ma - mi
    out = out * diff .+ mi
    return out
end

"""
    findsplit!(n, data[, splitval_cnt])

Find the best feature and its val for the given `node`.

If `splitval_cnt` is unspecified, value of 10 is used.
"""
function findsplit!(n::Node, data, splitval_cnt::Integer=10)
    setprediction!(n, data)
    bestscore = evaluate(n, data)
    bestftr, bestval = -1, 0.0
    for ftr = 1:length(data[1])
        mi, ma = getextremas(ftr, n, data)
        for splitval = uniform(mi, ma, splitval_cnt)
            candidatescore = evaluatesplit(ftr, splitval, n, data)
            if candidatescore < bestscore
                bestscore = candidatescore
                bestftr = ftr
                bestval = splitval
            end
        end
    end
    n.ftr = bestftr
    n.splitval = bestval
end

"""
    build!(parent, tree)

Recursively build tree from node.

In the first call `parent` is likely the root of the tree.
`tree` is passed because it holds attributes modifiing how should the build
proceed. 
Theoretically `parent` could belong to a different tree and `tree` would serve
as a recipe defining how should the build proceed.
However this is descouraged.
"""
function build!(parent::Node, tree)
    data = tree.data
    if stopdividing(parent, tree)
        parent.isleaf = true
        setprediction!(parent, data)
    else
        findsplit!(parent, data)
        if parent.ftr != -1
            parent.left, parent.right = splitnode(parent.ftr, parent.splitval, 
                                                parent, data)
            build!(parent.left, tree)
            build!(parent.right, tree)
        else
            # This happens when node attemps to split but the criterion is
            # minimized when unsplit. 
            # I don't know if this is still happening after the last rework
            # of criteria. TODO check this.
            parent.isleaf = true
        end
    end
end

"""
    buildtree!(tree)

Given `tree` build it.

`tree` should contain all specs needed to build it.
Works by recreating the `tree.root` and building with `build!`.
"""
function buildtree!(tree)
    tree.root = typeof(tree.root)(Vector(1:length(tree.data)))
    build!(tree.root, tree)
end

"""
    predict(datapoint, model)
    predict(datapoints, model)
"""
function predict(datapoint, tree)
    predict(datapoint, tree.root)
end

function predict(datapoint, node::Node)
    if node.isleaf
        return node.pred
    end
    if datapoint[node.ftr] < node.splitval
        return predict(datapoint, node.left)
    else
        return predict(datapoint, node.right)
    end
end

function predictall(datapoints, tree)
    predictall(datapoints, tree.root)
end

function predictall(datapoints, node::Node)
    predictions = []
    for dp=datapoints
        push!(predictions, predict(dp, node))
    end
    return predictions
end


# Broken
function printtree(n, space)
    if isnothing(n)
        return nothing
    end
    space += 1
    printtree(n.right, space)
    print("\n")
    for i=1:space
        print("    ")
    end
    print(n.pred)
    printtree(n.left, space)
end

