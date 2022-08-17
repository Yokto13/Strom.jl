include("../node.jl")
include("../utils.jl")
include("../randomforest.jl")

using Test

@testset "predict()" begin
    data = Data([
            Dato([1,2,3], 3),
            Dato([1,2,2], 1),
            Dato([7.5,0,2], 2),
            Dato([10,2,3], 2),
            Dato([8,2,3], 2),
            Dato([0,2,1], 1),
            Dato([5,2,3], 3),
           ], 3)
    treecnt = 200
    forest = RandomForest(data, treecnt, GiniNode())
    buildforest!(forest)
    for d=data
        @test d.y == argmax(predict(d, forest))
        @test d.y == argmax(predict(d.x, forest))
    end
    @test length(predict([1,1,1], forest)) == data.classcnt
end

@testset "predictall()" begin
    data = Data([
            Dato([1,2,3], 3),
            Dato([1,2,2], 1),
            Dato([7.5,0,2], 2),
            Dato([10,2,3], 2),
            Dato([8,2,3], 2),
            Dato([0,2,1], 1),
            Dato([5,2,3], 3),
           ], 3)
    treecnt = 200
    forest = RandomForest(data, treecnt, EntropyNode())
    buildforest!(forest)
    res = predictall(data, forest)
    @test length(res) == length(data)
    for i=1:length(res)
        @test argmax(res[i]) == data[i].y
    end
end

@testset "createtrees!()" begin
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
    forest = RandomForest(data, treecnt, GiniNode())
    createtrees!(forest)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test (forest.trees[i].data !== forest.trees[1].data)
    end
end


@testset "buildforest!()" begin
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

    forest = RandomForest(data, treecnt, GiniNode())
    buildforest!(forest)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test !isnothing(forest.trees[i].root)
        predict(data[1], forest.trees[i])
        @test (forest.trees[i].data !== forest.trees[1].data)
    end
    
    buildforest!(forest, true)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test !isnothing(forest.trees[i].root)
        predict(data[1], forest.trees[i])
        @test (forest.trees[i].data !== forest.trees[1].data)
    end
    
    forest = RandomForest(data, treecnt, GiniNode())
    forest.ftrsubset = (x -> Integer(ceil(log(x))))
    buildforest!(forest, true)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test !isnothing(forest.trees[i].root)
        predict(data[1], forest.trees[i])
        @test (forest.trees[i].data !== forest.trees[1].data)
    end
    
    forest = RandomForest(data, treecnt, GiniNode())
    forest.datasetsubset = (x -> Integer(ceil(0.5 * x)))
    buildforest!(forest, true)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test !isnothing(forest.trees[i].root)
        predict(data[1], forest.trees[i])
        @test (forest.trees[i].data !== forest.trees[1].data)
        @test (length(forest.trees[i].data) < length(forest.data))
    end

    forest = RandomForest(data, treecnt, GiniNode())
    buildforest!(forest, true)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test !isnothing(forest.trees[i].root)
        predict(data[1], forest.trees[i])
        @test (forest.trees[i].data !== forest.trees[1].data)
        @test (forest.trees[i].minnode == forest.minnode)
        @test (forest.trees[i].maxdepth == forest.maxdepth)
    end
    
    forest = RandomForest(data, treecnt, GiniNode())
    buildforest!(forest, false)
    @test length(forest.trees) == 0
    for i=2:treecnt 
        @test_throws BoundsError predict(data[1], forest.trees[i])
    end

    forest = RandomForest(data, treecnt, EntropyNode())
    buildforest!(forest)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test !isnothing(forest.trees[i].root)
        predict(data[1], forest.trees[i])
        @test (forest.trees[i].data !== forest.trees[1].data)
    end

    forest = RandomForest(data, treecnt, RegNode())
    buildforest!(forest)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test !isnothing(forest.trees[i].root)
        predict(data[1], forest.trees[i])
        @test (forest.trees[i].data !== forest.trees[1].data)
    end
end

