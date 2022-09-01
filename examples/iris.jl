using MLDatasets
using DataFrames

include("../src/Strom.jl")
using .Strom

println("Loading data.")

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

println("Data loaded.")

trees = [
        ClsTree(train, 1, 10000),
        ClsTree(train, 10, 10000),
        ClsTree(train, 1, 100),
        ClsTree(train, 10, 100),
        ClsTree(train, 1, 10),
        ClsTree(train, 10, 10),
        ClsTree(train, 1, 5),
        ClsTree(train, 10, 5),
        ClsTree(train, 1, 3),
        ClsTree(train, 10, 3),
        ClsTree(train, 1, 2),
        ClsTree(train, 10, 2),
        ClsTree(train, 60, 2),
       ]

println("Trees created.")

forests = [
        RandomForest(train, 5, GiniNode()),
        RandomForest(train, 10, GiniNode()),
        RandomForest(train, 300, GiniNode()),
        BoostForest(train, 30, ClsBoostNode()),
        BoostForest(train, 50, ClsBoostNode()),
          ]

println("Forests created.")

for t=trees
    buildtree!(t)
end

println("Trees trained.")

for f=forests
    buildforest!(f)
end

println("Forests trained.")

println("Prediction phase...")
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

println("Forest predictions.")

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
