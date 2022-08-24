include("utils.jl")

abstract type Node end

"""
    evaluate(left, right, data)

Calculate error for left and right and get its sum.

Used to score how good a certain node split is.
"""
function evaluate(left::Node, right::Node, data)
    return evaluate(left, data) + evaluate(right, data)
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


