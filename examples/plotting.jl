using MLDatasets
using DataFrames

include("../src/Strom.jl")
using .Strom

data = Data([
        Dato([1,2,3], 3),
        Dato([1,2,2], 1),
        Dato([7.5,0,2], 2),
        Dato([10,2,3], 2),
        Dato([8,2,3], 2),
        Dato([0,2,1], 1),
        Dato([5,2,3], 3),
        ], 3)
treecnt = 10
forest = BoostForest(data, treecnt, RegBoostNode())
println("Creating plots of loss durring boosted forest training.")
boostloss(forest, forest.data, "boostplot.png")
boostloss(forest, forest.data, "boostplotscatter.png"; seriestype = :scatter)

println("Difference between train/test loss for pet dataset.")
boosttraintest(forest, Data(data[1:4], 3), Data(data[5:end], 3), 
               "boosttraintest_simple.png")

dataset = BostonHousing()

data = []
for i=1:506
    row = Array(dataset.dataframe[i, :])
    push!(data, Dato(row[1:13], row[14]))
end

train = Data(data[1:400], 1)
test = Data(data[401:506], 1)

println("Difference between train/test loss for BostonHousing.")
forest = BoostForest(train, treecnt, RegBoostNode())
boosttraintest(forest, train, test, 
               "boosttraintest_boston.png")

data2d = Data([
            Dato([0, 0,], 2),
            Dato([1, 1,], 2),
            Dato([0, 1,], 1),
            Dato([1, 0,], 1),
              ], 2)

ct = ClsTree(data2d)
buildtree!(ct)
println("Creating a contour XOR plot.")
treecontour(ct, 0:0.05:1, 0:0.05:1, "xorcontourfill.png", fill=true)
println("Creating a tree plot.")
printtree(ct)
