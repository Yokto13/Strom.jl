include("../node.jl")
include("../utils.jl")

using Test

eps = 1e-7

@testset "uniform()" begin
    @test length(uniform(-1,2.3, 10)) == 10
    @test typeof(uniform(-1,2.3, 3)[1]) == Float64
    mi, ma = getextremas(uniform(2, 10, 100))
    @test mi >= 2
    @test ma <= 10
end

