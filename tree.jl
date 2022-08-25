using Random 

rng_seed = 42
Random.seed!(rng_seed)

include("regnode.jl")
include("clsnode.jl")
include("abstracttree.jl")

mutable struct Tree <: AbstractTree
    data
    root::Node
    minnode
    maxdepth
    ftrsubset
    # G, H are for boosting, shouldn't be set manually
    G::Union{Vector, Nothing}
    H::Union{Vector, Nothing}
end

RegTree(data) = Tree(data, RegNode(), 1, nothing, 1.0, nothing, nothing)
RegTree(data, minnode, maxdepth) = Tree(data, RegNode(), minnode, maxdepth, 1.0,
                                       nothing, nothing)
RegTree(data, ftrsubset) = Tree(data, RegNode(), 1, nothing, ftrsubset,
                                nothing, nothing)
RegTree(data, minnode, maxdepth, ftrsubset) = Tree(data, RegNode(), minnode, 
                                                   maxdepth, ftrsubset, nothing,
                                                  nothing)

ClsTree(data) = Tree(data, GiniNode(), 1, nothing, 1.0, nothing, nothing)
ClsTree(data, minnode, maxdepth) = Tree(data, GiniNode(), minnode, maxdepth, 1.0,
                                       nothing, nothing)
ClsTree(data, ftrsubset) = Tree(data, GiniNode(), 1, nothing, ftrsubset, nothing,
                               nothing)
ClsTree(data, minnode, maxdepth, ftrsubset) = Tree(data, GiniNode(), minnode, 
                                                   maxdepth, ftrsubset,
                                                  nothing, nothing)

Tree(data, root, minnode, maxdepth) = Tree(data, root, minnode, maxdepth, 1.0,
                                          nothing, nothing)

Tree(data, root, minnode, maxdepth, G, H) = Tree(data, root, minnode, maxdepth, 1.0,
                                          G, H)

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
    splitnode(ftr, val, node, data)

Get child-nodes containing `datainds` split on feature `ftr` and value `val`.

All points < `val` are placed to the `left`, the rest to the `right`. 
"""
function splitnode(ftr, val, n::Node, data)
    left, right = typeof(n)(), typeof(n)()
    left.depth = right.depth = n.depth + 1
    left.id = n.id * 2
    right.id = n.id * 2 + 1
    for j = 1:length(n.datainds)
        if data[n.datainds[j]][ftr] < val
            push!(left.datainds, n.datainds[j])
        else
            push!(right.datainds, n.datainds[j])
        end
    end
    @assert length(right.datainds) != 0
    @assert length(left.datainds) != 0
    return (left, right)
end

"""
    evaluatesplit(ftr, val, n, tree)
"""
function evaluatesplit(ftr::Int64, val::Float64, n::Node, tree)
    left, right = splitnode(ftr, val, n, tree.data)
    setprediction!(left, tree)
    setprediction!(right, tree)
    # return - evaluate(left, right, tree) + evaluate(n, tree)
    return evaluate(left, right, tree) - evaluate(n, tree)
    # return evaluate(left, right, tree)
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

function skipftr(ftrsubset)
    return ftrsubset < rand()
end

"""
    findsplit!(n, tree, ftrsubset)

Find the best feature and its val for the given `node`.

# Arguments
- `ftrsubset`: the franction of features to split on. For single tree
    you probably want 1.0. For forests this is a hyperparameter to be tuned.
    √ of the number of features is usally a good idea, remember it needs fraction.
"""
function findsplit!(n::Node, tree, ftrsubset=1.0)
    setprediction!(n, tree)
    bestscore = Inf
    #printl("bestscore ", bestscore)
    bestftr, bestval = -1, 0.0
    for ftr = 1:length(tree.data[1])
        if skipftr(ftrsubset) && continue end
        mi, ma = getextremas(ftr, n, tree.data)
        # for splitval = uniform(mi, ma, splitval_cnt)
        sorted = sort([tree.data[i] for i in n.datainds], by=x -> x[ftr])
        for i = 2:length(sorted) 
            if sorted[i][ftr] == sorted[i - 1][ftr] && continue end
            splitval = (sorted[i][ftr] + sorted[i - 1][ftr]) / 2
            #println(ftr)
            #println(splitval)
            #println(sorted)
            candidatescore = evaluatesplit(ftr, splitval, n, tree)
            #println("candidatescore ", candidatescore)
            if candidatescore < bestscore
                bestscore = candidatescore
                bestftr = ftr
                bestval = splitval
            end
        end
    end
    n.ftr = bestftr
    n.splitval = bestval
    #println(bestftr, " ", bestval)
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
        setprediction!(parent, tree)
    else
        findsplit!(parent, tree, tree.ftrsubset)
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
            setprediction!(parent, tree)
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
function predict(datapoint, tree::Tree)
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

function predictall(datapoints, tree::Tree)
    predictall(datapoints, tree.root)
end

function predictall(datapoints, node::Node)
    predictions = []
    for dp=datapoints
        push!(predictions, predict(dp, node))
    end
    return predictions
end


# Broke#n
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

