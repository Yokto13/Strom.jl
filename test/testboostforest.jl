include("../src/node.jl")
include("../src/utils.jl")
include("../src/boostnode.jl")
include("../src/regboostnode.jl")
include("../src/clsboostnode.jl")
include("../src/boostforest.jl")
include("../src/abstracttree.jl")
include("../src/tree.jl")

using Test

eps = 1e-7

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
    forest = BoostForest(data, treecnt, RegBoostNode())
    updateH!(forest)
    @test forest.H == ones(length(data))
    # --------------------- cls:
    treecnt = 200
    forest = BoostForest(data, treecnt, ClsBoostNode())
    initpreds!(forest)
    updateH!(forest, 2)
    @test length(forest.H) == length(data)
    forest = BoostForest(Data(data[1:3], 3), treecnt, ClsBoostNode())
    initpreds!(forest)
    forest.logits = [1 0.5 0; 0.5 0.5 0; 0 0.5 10]
    sf = softmax(forest.logits[:, 1])
    updateH!(forest, 1)
    @test length(forest.H) == 3
    @test forest.H ≈ sf .* (1 .- sf)
end

function cse(pvec, correct_cls)
    return - log(pvec[correct_cls])
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
    treecnt = 200
    forest = BoostForest(data, treecnt, RegBoostNode())
    losses = buildforest!(forest, evloss=(x, y) -> (x - y) ^ 2)
    for i=1:length(losses) - 1
        @test losses[i] ≥ losses[i + 1]
    end
    for d=data
        @test d.y == round(predict(d, forest))
    end
    @test evaluate(data, forest, (x, y) -> (x - y) ^ 2) - eps < 0
    treecnt = 200
    forest = BoostForest(data, treecnt, ClsBoostNode())
    losses = buildforest!(forest, evloss=cse)
    for i=1:length(losses) - 1
        @test losses[i] ≥ losses[i + 1]
    end
    for d=data
        @test d.y == argmax(predict(d, forest))
    end
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
