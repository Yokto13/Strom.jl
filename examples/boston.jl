using MLDatasets
using DataFrames

include("../src/Strom.jl")
using .Strom

println("Loading data.")

dataset = BostonHousing()

data = []
for i=1:506
    row = Array(dataset.dataframe[i, :])
    push!(data, Dato(row[1:13], row[14]))
end

train = data[1:100]
test = data[401:506]

println("Data loaded.")

rt1 = RegTree(train, 2, 10000); buildtree!(rt1)
println("Tree 1 built. ",rt1.minnode)
rt2 = RegTree(train, 10, 10000); buildtree!(rt2)
println("Tree 2 built. ",rt2.minnode)
rt3 = RegTree(train, 20, 10000); buildtree!(rt3)
println("Tree 3 built. ",rt3.minnode)
rt4 = RegTree(train, 100, 10000); buildtree!(rt4)
println("Tree 4 built. ",rt4.minnode)

bs = []
for i=1:8
    push!(bs, BoostForest(Data(train, 1), 2 ^ i, RegBoostNode()))
    buildforest!(bs[i])
    println("Boost forest with ", 2 ^ i, "trees built.")
    println("Boost ", i,  " loss on train ", MSE_all(Data(train, 1), bs[i]))
    println("Boost ", i, " loss on test ", MSE_all(Data(test, 1), bs[i]))
end

println("Tree 1 loss on train ", MSE_all(train, rt1))
println("Tree 1 loss on test ", MSE_all(test, rt1))

println("Tree 2 loss on train ", MSE_all(train, rt2))
println("Tree 2 loss on test ", MSE_all(test, rt2))

println("Tree 3 loss on train ", MSE_all(train, rt3))
println("Tree 3 loss on test ", MSE_all(test, rt3))

println("Tree 4 loss on train ", MSE_all(train, rt4))
println("Tree 4 loss on test ", MSE_all(test, rt4))
