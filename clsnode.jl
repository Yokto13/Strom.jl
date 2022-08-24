include("node.jl")

abstract type ClsNode <: Node end

mutable struct GiniNode <: ClsNode
    isleaf::Bool
    pred::Vector
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
    id::Integer
end

GiniNode(v::Vector{Int64}) = GiniNode(false, [], v, -1, -1, nothing, nothing, 1, 1)
GiniNode() = GiniNode(false, [], [], -1, -1, nothing, nothing, 1, 1)

mutable struct EntropyNode <: ClsNode
    isleaf::Bool
    pred::Vector
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
    id::Integer
end

EntropyNode(v::Vector{Int64}) = EntropyNode(false, [], v, -1, -1, nothing,
                                            nothing, 1, 1)
EntropyNode() = EntropyNode(false, [], [], -1, -1, nothing, nothing, 1, 1)

function calcprediction(n::ClsNode, tree)
    data = tree.data
    pred = zeros(data.classcnt)
    sz = length(n.datainds)
    for j = 1:sz
        i = n.datainds[j]
        pred[data[i].y] += 1
    end
    return pred / length(n.datainds)
end

"""
    evaluate(n, tree)

Evaluate node.
"""
function evaluate(n::GiniNode, tree)
    return gini(n, tree.data)
end

function evaluate(n::EntropyNode, tree)
    return entropy(n, tree.data)
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

