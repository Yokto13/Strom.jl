include("node.jl")
include("abstracttree.jl")

mutable struct RegNode <: Node
    isleaf::Bool
    pred::Numeric64
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
    id::Integer
end

RegNode(v::Vector{Int64}) = RegNode(false, -1, v, -1, -1, nothing, nothing, 1, 1)
RegNode() = RegNode(false, -1, [], -1, -1, nothing, nothing, 1, 1)

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
    return pred / length(n.datainds)
end

function calcprediction(n::RegNode, tree::AbstractTree)
    data = tree.data
    calcprediction(n, data)
end

"""
    evaluate(n, tree)

Evaluate node.
"""
function evaluate(n::RegNode, tree)
    return SMSE(n, tree.data)
end

"""
    SMSE(n, data)

Sum of mean squared errors belonging to `n.datainds`,
"""
function SMSE(n::RegNode, data)
    sz = length(n.datainds)
    res::Float64 = 0.0
    for j = 1:sz
        i = n.datainds[j]
        res += (n.pred - data[i].y)^2
    end
    return res
end
