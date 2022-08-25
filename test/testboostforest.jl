include("../src/node.jl")
include("../src/utils.jl")
include("../src/boostforest.jl")
include("../src/regboostnode.jl")
include("../src/clsboostnode.jl")

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
    treecnt = 127
    forest = BoostForest(data, treecnt, ClsBoostNode())
    initpreds!(forest)
    updateG!(forest, 1)
    @test length(forest.G) == length(data)
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
    println("11")
    forest = BoostForest(data, treecnt, RegBoostNode())
    println("111")
    updateH!(forest)
    println("111")
    @test forest.H == ones(length(data))
    # --------------------- cls:
    treecnt = 200
    forest = BoostForest(data, treecnt, ClsBoostNode())
    println("111")
    println("111??")
    initpreds!(forest)
    println("112")
    updateH!(forest, 2)
    println("112")
    @test length(forest.H) == length(data)
    print("C")
    forest = BoostForest(Data(data[1:3], 3), treecnt, ClsBoostNode())
    initpreds!(forest)
    forest.logits = [1 0.5 0; 0.5 0.5 0; 0 0.5 10]
    sf = softmax(forest.logits[:, 1])
    println(sf)
    updateH!(forest, 1)
    print("B")
    @test length(forest.H) == 3
    @test forest.H ≈ sf .* (1 .- sf)
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
    treecnt = 20
    forest = BoostForest(data, treecnt, RegBoostNode())
    buildforest!(forest)
    for d=data
        @test d.y == round(predict(d, forest))
    end
    treecnt = 4
    #forest = BoostForest(data, treecnt, ClsBoostNode())
    #buildforest!(forest)
    #for d=data
    #    println(predict(d, forest), " ", d.y)
    #    @test d.y == argmax(predict(d, forest))
    #end
end

@testset "learningrate" begin
    data = Data([
            Dato([1,2,3], 3),
            Dato([1,2,2], 1),
            Dato([7.5,0,2], 2),
            Dato([10,2,3], 2),
            Dato([8,2,3], 2),
            Dato([0,2,1], 1),
            Dato([5,2,3], 3),
           ], 3)
    treecnt = 3
    forest = BoostForest(data, treecnt, RegBoostNode())
    forest.α = 0.01
    buildforest!(forest)
    res = predictall(data, forest)
    loss1 = 0
    @test length(res) == length(data)
    for i=1:length(res)
        loss1 += (res[i] - data[i].y)^2
    end
    forest = BoostForest(data, treecnt, RegBoostNode())
    forest.α = 0.1
    buildforest!(forest)
    res = predictall(data, forest)
    loss2 = 0
    @test length(res) == length(data)
    for i=1:length(res)
        loss2 += (res[i] - data[i].y)^2
    end
    @test loss1 > loss2
    forest = BoostForest(data, treecnt, RegBoostNode())
    forest.α = 0.5
    buildforest!(forest)
    res = predictall(data, forest)
    loss3 = 0
    @test length(res) == length(data)
    for i=1:length(res)
        loss3 += (res[i] - data[i].y)^2
    end
    @test loss2 > loss3
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

@testset "buildforest!()" begin
    treecnt = 10
    classcnt = 3
    data = Data([
            Dato([1,2,3], 3),
            Dato([1,2,2], 1),
            Dato([7.5,0,2], 2),
            Dato([10,2,3], 2),
            Dato([8,2,3], 2),
            Dato([0,2,1], 1),
            Dato([5,2,3], 3),
           ], classcnt)
    forest = BoostForest(data, treecnt, RegBoostNode())
    #buildforest!(forest)
    #@test length(forest.trees) == treecnt
    forest = BoostForest(data, treecnt, ClsBoostNode())
    buildforest!(forest)
    @test size(forest.trees) == (treecnt,)
    @test forest.treestrained == treecnt
end
