using Random

rng_seed = 42
Random.seed!(rng_seed)

include("utils.jl")

abstract type Node end
abstract type Tree end

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

mutable struct ClsNode <: Node
    isleaf::Bool
    pred::Numeric64
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
end

ClsNode(v::Vector{Int64}) = ClsNode(false, -1, v, -1, -1, nothing, nothing, 1)
ClsNode() = ClsNode(false, -1, [], -1, -1, nothing, nothing, 1)

mutable struct RegTree <: Tree
    data::Vector{Dato}
    root::RegNode
    minnode
    maxdepth
end

RegTree(data) = RegTree(data, RegNode(), 2, nothing)
RegTree(data, minnode, maxdepth) = RegTree(data, RegNode(), minnode, maxdepth)

"""
    setprediction!(n, data)

Set prediction to node `n` based on `n.datainds`
"""
function setprediction!(n::RegNode, data)
    result::Float64 = 0.0
    sz = length(n.datainds)
    for j = 1:sz
        i = n.datainds[j]
        result += data[i].y
    end
    result/= sz
    n.pred = result
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
    return stopdividing(n, cond)
end

"""
    get_stopcondition(tree)

Creates partial function from `tree` which can determine node split.
"""
function get_stopcondition(tree::Tree)
    return x -> ((!isnothing(tree.maxdepth) && n.depth >= tree.maxdepth) 
                 || (!isnothing(tree.minnode) && length(n.datainds) < tree.minnode)
                )
end

"""
    evaluate(n)

Evaluate node.
"""
function evaluate(n::RegNode)
    return MSE(n)
end
"""
    evaluate(left, right)

Calculate error for left and right and get its sum.

Used to score how good a certain node split is.
"""
function evaluate(left::RegNode, right::RegNode)
    return evaluate(left) + evaluate(right)
end

function MSE(n::Node)
    sz = length(n.datainds)
    res::Float64 = 0.0
    for j = 1:sz
        i = n.datainds[j]
        res += (n.pred - data[i].y)^2
    end
    if sz != 0
        res /= sz
    end
    return res
end

"""
    splitnode(ftr, val, node, data)

Get child-nodes containing `datainds` split on feature `ftr` and value `val`.

All points < `val` are placed to the `left`, the rest to the `right`. 
"""
function splitnode(ftr, val, n::Node, data)
    left, right = typeof(n)(), typeof(n)()
    for j = 1:length(n.datainds)
        #println(j, " ", ftr)
        if data[n.datainds[j]][ftr] < val
            push!(left.datainds, n.datainds[j])
        else
            push!(right.datainds, n.datainds[j])
        end
    end
    return (left, right)
end

"""
    eva;iatesplit(ftr, val, n, data)
"""
function evaluatesplit(ftr::Int64, val::Float64, n::Node, data)
    left, right = splitnode(ftr, val, n, data)
    setprediction!(left, data)
    setprediction!(right, data)
    return evaluate(left, right)
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
    #println("ftr ",ftr) 
    #println(data)
    #println(typeof(data))
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

If `splitval_cnt` is unspecified, value of 50 is used.
"""
function findsplit!(n::Node, data, splitval_cnt::Integer=50)
    setprediction!(n, data)
    bestscore = evaluate(n)
    bestftr, bestval = -2, 0.0
    for ftr = 1:length(data[1])
        mi, ma = getextremas(ftr, n, data)
        for splitval = uniform(mi, ma, splitval_cnt)
            splitval = (ma + mi) / 2
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
        if parent.ftr != -2
            parent.left, parent.right = splitnode(parent.ftr, parent.splitval, 
                                                parent, data)
            build!(parent.left, tree)
            build!(parent.right, tree)
        else
            parent.ftr = -1
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

