include("utils.jl")
include("node.jl")

mutable struct RegTree
    root::RegNode
    data::Vector{Dato}
end

function setprediction(n::RegNode, data)
    result::Float64 = 0.0
    sz = length(n.examples_indices)
    for j = 1:sz
        i = n.examples_indices[j]
        result += data[i].y
    end
    result/= sz
    n.pred = result
end

function stopdividing(n::Node)
    if length(n.examples_indices) < 3
        return true
    end
    return false
end

function evaluate(left::RegNode, right::RegNode)
    return MSE(left) + MSE(right)
end

function MSE(n::Node)
    sz = length(n.examples_indices)
    res::Float64 = 0.0
    for j = 1:sz
        i = n.examples_indices[j]
        res += (n.pred - data[i].y)^2
    end
    res /= sz
    return res
end

function splitnode(ftr, val, n::Node)
    # Make sure children are the same type as parrent.
    left = typeof(n)()
    right = typeof(n)()
    ex_sz = length(n.examples_indices)
    for j = 1:ex_sz
        if data[n.examples_indices[j]][ftr] < val
            push!(left.examples_indices, n.examples_indices[j])
        else
            push!(right.examples_indices, n.examples_indices[j])
        end
    end
    return (left, right)
end

function evaluatesplit(ftr::Int64, val::Float64, n::Node)
    left, right = splitnode(ftr, val, n)
    return evaluate(left, right)
end

function getextremas(ftr::Int64, ex_sz::Int64, n::Node)
    mi::Float64 = Inf
    ma::Float64 = -Inf
    for j = 1:ex_sz
        ex_i = n.examples_indices[j]
        mi = min(mi, data[ex_i][ftr])
        ma = max(ma, data[ex_i][ftr])
    end
    @assert(mi != Inf)
    @assert(ma != -Inf)
    return (mi, ma)
end

function findsplit!(n::Node)
    ftr_sz = length(data[1])
    ex_sz = length(n.examples_indices)
    bestscore = Inf
    bestftr = -1
    bestval = 0.0
    for ftr = 1:ftr_sz
        mi, ma = getextremas(ftr, ex_sz, n)
        # Later iterate over more split vals and test different ways to get
        # vals
        splitval = (ma + mi) / 2.0
        score_candidate = evaluatesplit(ftr, splitval, n)
        # Nelze se nějak vyhýbat floatům? Vždyť se tu to v nich topí.
        if score_candidate < bestscore
            bestscore = score_candidate
            bestftr = ftr
            bestval = splitval
        end
    end
    @assert(bestftr != -1)
    n.ftr = bestftr
    n.splitval = bestval
end

function build(parent::Node, data)
    if stopdividing(parent)
        parent.isleaf = true
        setprediction(parent, data)
    else
        findsplit!(parent)
        parent.left, parent.right = splitnode(parent.ftr, parent.splitval, 
                                              parent)
        build(parent.left, data)
        build(parent.right, data)
    end
end

function buildtree!(tree)
    tree.root = typeof(tree.root)(Vector(1:length(tree.data)))
    build(tree.root, tree.data)
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

function predict(datapoint, tree)
    predict(datapoint, tree.root)
end

data = [
        Dato([1,2,3], 0)
        Dato([1,2,2], 1)
        Dato([1,0,2], 1)
        Dato([10,2,3], 0)
       ]

rt = RegTree(RegNode(), data)
buildtree!(rt)
