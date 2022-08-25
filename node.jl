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
    setprediction!(n, dataobject)

Calculates prediction and sets it to `n.pred`.
# Arguments
- `n`: node to wchich the prediction will be set.
- `dataobject`: tree, Data, array.
Something holding the training data.
`Tree` is passed here usually if we are dealing with boosting.
"""
function setprediction!(n, dataobject)
    pred = calcprediction(n, dataobject)
    n.pred = pred
end


