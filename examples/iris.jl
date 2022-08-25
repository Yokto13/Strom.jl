using MLDatasets
using DataFrames

include("../src/utils.jl")
include("../src/node.jl")
include("../src/randomforest.jl")
include("../src/boostforest.jl")

dataset = Iris()

data = []


name_id = Dict([
                ("Iris-setosa", 1),
                ("Iris-versicolor", 2),
                ("Iris-virginica", 3),
               ])

id_name = Dict([
                (1, "Iris-setosa"),
                (2, "Iris-versicolor"),
                (3, "Iris-virginica"),
               ])

targets = dataset.targets

for i=1:150
    row = Array(dataset.features[i, :])
    push!(data, Dato(row[1:4], name_id[targets[!, 1][i]]))
end


shuffle!(data)

train = data[1:135]
test = data[136:150]

train = Data(train, 3)
test =  Data(test, 3)

#println(train)
#println(test)

trees = [
        ClsTree(train, 1, 10000),
        ClsTree(train, 3, 10000),
        ClsTree(train, 5, 10000),
        ClsTree(train, 10, 10000),
        ClsTree(train, 20, 10000),
        ClsTree(train, 1, 100),
        ClsTree(train, 3, 100),
        ClsTree(train, 5, 100),
        ClsTree(train, 10, 100),
        ClsTree(train, 20, 100),
        ClsTree(train, 1, 10),
        ClsTree(train, 3, 10),
        ClsTree(train, 5, 10),
        ClsTree(train, 10, 10),
        ClsTree(train, 20, 10),
        ClsTree(train, 1, 5),
        ClsTree(train, 3, 5),
        ClsTree(train, 5, 5),
        ClsTree(train, 10, 5),
        ClsTree(train, 20, 5),
        ClsTree(train, 1, 3),
        ClsTree(train, 3, 3),
        ClsTree(train, 5, 3),
        ClsTree(train, 10, 3),
        ClsTree(train, 20, 3),
        ClsTree(train, 20, 2),
        ClsTree(train, 1, 2),
        ClsTree(train, 3, 2),
        ClsTree(train, 5, 2),
        ClsTree(train, 10, 2),
        ClsTree(train, 20, 2),
        ClsTree(train, 40, 2),
        ClsTree(train, 60, 2),
       ]

forests = [
        RandomForest(train, 5, GiniNode()),
        RandomForest(train, 10, GiniNode()),
        RandomForest(train, 50, GiniNode()),
        RandomForest(train, 100, GiniNode()),
        RandomForest(train, 300, GiniNode()),
        RandomForest(train, 3000, GiniNode(), 3, 3),
          ]

for t=trees
    buildtree!(t)
end

for f=forests
    buildforest!(f)
end
for t=trees
    println(t.minnode, " ", t.maxdepth)
    prds = predictall(train, t)
    prds = argmaxx(prds)
    pairs = []
    for i=1:length(train)
        push!(pairs,(prds[i], train[i].y))
    end
    println(accuracy(pairs))

    prds = predictall(test, t)
    prds = argmaxx(prds)
    pairs = []
    for i=1:length(test)
        push!(pairs,(prds[i], test[i].y))
    end
    println(accuracy(pairs))
    println("-----------")
end

for t=forests
    println(t.minnode, " ", t.maxdepth)
    prds = predictall(train, t)
    println(prds[1])
    prds = argmaxx(prds)
    pairs = []
    for i=1:length(train)
        push!(pairs,(prds[i], train[i].y))
    end
    println(accuracy(pairs))

    prds = predictall(test, t)
    println(prds[1])
    prds = argmaxx(prds)
    pairs = []
    for i=1:length(test)
        push!(pairs,(prds[i], test[i].y))
    end
    println(accuracy(pairs))
    println("-----------")
end
