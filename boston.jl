using MLDatasets
using DataFrames

include("utils.jl")
include("node.jl")

dataset = BostonHousing()

data = []

for i=1:506
    row = Array(dataset.dataframe[i, :])
    push!(data, Dato(row[1:13], row[14]))
end

train = data[1:400]
test = data[401:506]

function MSE(y_hat, y)
    return (y_hat - y) ^ 2 
end

function MSE_all(data, tree)
    all = 0
    for d=data
        all += MSE(predict(d.x, tree), d.y)
    end
    all
end

rt1 = RegTree(train, x -> length(x.datainds) < 2); buildtree!(rt1)
println("Tree 1 built.")
rt2 = RegTree(train, x -> length(x.datainds) < 10); buildtree!(rt2)
println("Tree 2 built.")
rt3 = RegTree(train, x -> length(x.datainds) < 20); buildtree!(rt3)
println("Tree 3 built.")
rt4 = RegTree(train, x -> length(x.datainds) < 100); buildtree!(rt4)
println("Tree 4 built.")

println("Tree 1 loss on train ", MSE_all(train, rt1))
println("Tree 1 loss on test ", MSE_all(test, rt1))

println("Tree 2 loss on train ", MSE_all(train, rt2))
println("Tree 2 loss on test ", MSE_all(test, rt2))

println("Tree 3 loss on train ", MSE_all(train, rt3))
println("Tree 3 loss on test ", MSE_all(test, rt3))

println("Tree 4 loss on train ", MSE_all(train, rt4))
println("Tree 4 loss on test ", MSE_all(test, rt4))
