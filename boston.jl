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

train = data[1:100]
test = data[401:506]

rt1 = RegTree(train, 2, 10000); buildtree!(rt1)
println("Tree 1 built. ",rt1.minnode)
rt2 = RegTree(train, 10, 10000); buildtree!(rt2)
println("Tree 2 built. ",rt2.minnode)
rt3 = RegTree(train, 20, 10000); buildtree!(rt3)
println("Tree 3 built. ",rt3.minnode)
rt4 = RegTree(train, 100, 10000); buildtree!(rt4)
println("Tree 4 built. ",rt4.minnode)

println("Tree 1 loss on train ", MSE_all(train, rt1))
println("Tree 1 loss on test ", MSE_all(test, rt1))

println("Tree 2 loss on train ", MSE_all(train, rt2))
println("Tree 2 loss on test ", MSE_all(test, rt2))

println("Tree 3 loss on train ", MSE_all(train, rt3))
println("Tree 3 loss on test ", MSE_all(test, rt3))

println("Tree 4 loss on train ", MSE_all(train, rt4))
println("Tree 4 loss on test ", MSE_all(test, rt4))
