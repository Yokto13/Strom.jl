include("../src/node.jl")
include("../src/abstracttree.jl")
include("../src/boostnode.jl")
include("../src/regboostnode.jl")
include("../src/clsboostnode.jl")
include("../src/tree.jl")

using Test

eps = 1e-7

# TODO tests with cls 
function test_criterion(f::Function)
    G = [1, -1]
    H = ones(2)
    n = RegBoostNode()
    @test isnan(f(n, G, H))
    n = RegBoostNode([1])
    @test f(n, G, H) == -0.5
    n = RegBoostNode([1, 2])
    @test f(n, G, H) == 0
    n = ClsBoostNode()
    @test isnan(f(n, G, H))
    n = ClsBoostNode([1])
    @test f(n, G, H) == -0.5
    n = ClsBoostNode([1, 2])
    @test f(n, G, H) == 0
end

@testset "similarity()" begin
    test_criterion(similarity)
end

@testset "evaluate()" begin
    evaluate2(n, G, H) = evaluate(n, Tree([], RegBoostNode(),1, 1, 1, G, H, 0))
    test_criterion(evaluate2)
end
