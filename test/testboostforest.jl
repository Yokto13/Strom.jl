include("../node.jl")
include("../utils.jl")
include("../boostforest.jl")
include("../regboostnode.jl")
# include("../clsboostnode.jl")

using Test

@testset "updateG!()" begin
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
    forest = BoostForest(data, treecnt, RegBoostNode())
    updateG!(forest)
    @test length(forest.G) == length(data)
    # TODO more tests
end

@testset "updateH!()" begin
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
    forest = BoostForest(data, treecnt, RegBoostNode())
    updateH!(forest)
    @test forest.H == ones(length(data))
end

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
    treecnt = 4
    forest = BoostForest(data, treecnt, RegBoostNode())
    buildforest!(forest)
    for d=data
        println(predict(d, forest), " ", d.y)
        @test d.y == round(predict(d, forest))
    end
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
    forest = BoostForest(data, treecnt, RegBoostNode())
    buildforest!(forest)
    res = predictall(data, forest)
    @test length(res) == length(data)
    for i=1:length(res)
        @test round(res[i]) == data[i].y
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
    forest = BoostForest(data, treecnt, RegBoostNode())
    createtrees!(forest)
    @test length(forest.trees) == treecnt
    for i=2:treecnt 
        @test (forest.trees[i].data !== forest.trees[1].data)
    end
end

# TODO buildforest test
