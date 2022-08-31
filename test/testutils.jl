include("../src/abstracttree.jl")
include("../src/node.jl")
include("../src/utils.jl")
include("../src/tree.jl")

using Test

eps = 1e-7

@testset "uniform()" begin
    @test length(uniform(-1,2.3, 10)) == 10
    @test typeof(uniform(-1,2.3, 3)[1]) == Float64
    mi, ma = getextremas(uniform(2, 10, 100))
    @test mi >= 2
    @test ma <= 10
end

@testset "softmax()" begin
    for i=1:5
        v = rand(100 * i)
        @test sum(softmax(v)) ≈ 1.0
    end
    o = softmax([0, 0, 1])
    @test o[1] == o[2]
    @test o[3] > o[2]
end
