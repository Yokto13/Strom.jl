include("../tree.jl")

using Test

eps = 1e-7

@testset "setprediction!()" begin
    # Regression
    data = [Dato([1], 1), Dato([1],1), Dato([2], 3)]
    n = RegNode([1,3])
    setprediction!(n, data)
    @test (n.pred - 2) < eps
    @test_throws Exception setprediction!(n, [])
    # Classification
    data = Data([Dato([1], 1), Dato([1],2), Dato([2], 3)], 3)
    n = GiniNode([1,3])
    setprediction!(n, data)
    @test (n.pred - [0.5, 0., 0.5]) < zeros(3) .+ eps
    @test_throws Exception setprediction!(n, [])
end

@testset "stopdividing()" begin
    n = EntropyNode([1,2,3,4])
    @test stopdividing(n, x -> length(x.datainds) < 4) == false
    n = EntropyNode([1,2])
    @test stopdividing(n, x -> length(x.datainds) < 4) == true
    @test stopdividing(n, x -> true) == true
    @test stopdividing(n, x -> false) == false
end

@testset "splitnode()" begin
    data = [Dato([1], 1), Dato([1],1), Dato([2], 3)]
    n = RegNode([1, 2, 3])
    l, r = splitnode(1, 2, n, data)
    @test length(l.datainds) == 2
    @test length(r.datainds) == 1
end

@testset "getextremas()" begin
    data = [Dato([-11], 1), Dato([1],1), Dato([2], 3)]
    n = GiniNode([1, 2, 3])
    @test getextremas(1, n, data) == (-11, 2)
    v = [1]
    @test getextremas(v) == (1, 1)
    push!(v, 2)
    @test getextremas(v) == (1, 2)
    push!(v, 3)
    @test getextremas(v) == (1, 3)
end

@testset "findsplit!()" begin
    data = [Dato([1], 1), Dato([1.1],1), Dato([2], 3)]
    n = RegNode([1, 2, 3])
    findsplit!(n, data, 1000)
    println("The best splitval is ", n.splitval)
    @test n.ftr == 1
    @test n.splitval > 1.1
    @test n.splitval < 2
end

@testset "get_stopcondition()" begin
    data = [Dato([1], 1), Dato([1.1],1), Dato([2], 3)]
    rt = RegTree(data, 2, 5)
    cond = get_stopcondition(rt)
    n = RegNode([1,3])
    n.depth = 4
    @test !cond(n)
    rt.maxdepth = 3
    @test cond(n)
    rt.maxdepth = 5
    rt.minnode = 4
    @test cond(n) 
    rt.maxdepth = 5
    rt.minnode = 1
    @test !cond(n) 
    rt.maxdepth = nothing
    rt.minnode = nothing
    @test !cond(n) 
    rt.maxdepth = 1
    rt.minnode = nothing
    @test cond(n) 
    rt.maxdepth = nothing
    rt.minnode = 10
    @test cond(n) 
end

@testset "build()" begin
    data = [
        Dato([1,2,3], 2),
        Dato([1,2,2], 1),
        Dato([1,0,2], 1),
        Dato([10,2,3], 2),
       ]
    data = Data(data, 2)
    rt = RegTree(data)
    buildtree!(rt)
    @test !isnothing(rt.root)
    t = ClsTree(data)
    buildtree!(t)
    @test !isnothing(t.root)
    t = ClsTree(data)
    t.root = EntropyNode()
    buildtree!(t)
    @test !isnothing(t.root)
    rt = RegTree(data)
    build!(rt.root, rt)
    @test !isnothing(rt.root)
    t = ClsTree(data)
    build!(t.root, t)
    @test !isnothing(t.root)
    t = ClsTree(data)
    t.root = EntropyNode()
    build!(t.root, t)
    @test !isnothing(t.root)
    n = GiniNode()
    build!(n, t)
    @test !isnothing(n)
end

@testset "predict()" begin
    data = [
        Dato([1,2,3], 0),
        Dato([1,2,2], 1),
        Dato([1,0,2], 1),
        Dato([10,2,3], 0),
       ]
    rt = RegTree(data)
    buildtree!(rt)
    for d=data
        p = predict(d, rt)
        @test p == predict(d, rt.root)
        @test p == d.y
    end
end
