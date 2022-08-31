include("../src/plot.jl")

using MLDatasets
using DataFrames

data = Data([
        Dato([1,2,3], 3),
        Dato([1,2,2], 1),
        Dato([7.5,0,2], 2),
        Dato([10,2,3], 2),
        Dato([8,2,3], 2),
        Dato([0,2,1], 1),
        Dato([5,2,3], 3),
        ], 3)
treecnt = 20
forest = BoostForest(data, treecnt, RegBoostNode())
#boostloss(forest, forest.data, "boostplot.png")
#boostloss(forest, forest.data, "boostplotscatter.png"; seriestype = :scatter)

#boosttraintest(forest, Data(data[1:4], 3), Data(data[5:end], 3), 
#               "boosttraintest_simple.png")

dataset = BostonHousing()

data = []
for i=1:506
    row = Array(dataset.dataframe[i, :])
    push!(data, Dato(row[1:13], row[14]))
end

#train = Data(data[1:400], 1)
#test = Data(data[401:506], 1)

#forest = BoostForest(train, treecnt, RegBoostNode())
#boosttraintest(forest, train, test, 
#               "boosttraintest_boston.png")

data2d = Data([
            Dato([0, 0,], 2),
            Dato([1, 1,], 2),
            Dato([0, 1,], 1),
            Dato([1, 0,], 1),
              ], 2)

ct = ClsTree(data2d)
buildtree!(ct)
for d=data2d
    println(predict(d, ct))
end
treecontour(ct, 0:0.05:1, 0:0.05:1, "xorcontourfill.png", fill=true)
printtree(ct)
