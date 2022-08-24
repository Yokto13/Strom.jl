include("../boostnode.jl")
include("../regboostnode.jl")

using Test

eps = 1e-7

# TODO tests with cls 
function test_criterion(f::Function)
    G = [1, -1]
    H = ones(2)
    n = RegBoostNode()
    @test isnan(f(n, G, H))
    n = RegBoostNode([1])
    @test f(n, G, H) == 1
    n = RegBoostNode([1, 2])
    @test f(n, G, H) == 0
    n = RegBoostNode([1], 1)
    @test f(n, G, H) == 0.5
    G = [10]
    n = RegBoostNode([1], 1)
    @test f(n, G, H) == 50
end

@testset "similarity()" begin
    test_criterion(similarity)
end

@testset "evaluate()" begin
    test_criterion(evaluate)
end
