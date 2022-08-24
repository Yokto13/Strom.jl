include("utils.jl")

abstract type Node end

"""
    evaluate(left, right, data)

Calculate error for left and right and get its sum.

Used to score how good a certain node split is.
"""
function evaluate(left::Node, right::Node, tree)
    return evaluate(left, tree) + evaluate(right, tree)
end

"""
    setprediction!(n, data)

Calculates prediction and sets it to `n.pred`.
"""
function setprediction!(n, tree)
    pred = calcprediction(n, tree)
    n.pred = pred
end


