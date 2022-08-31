include("../src/utils.jl")
include("../src/abstracttree.jl")
include("../src/node.jl")
include("../src/regnode.jl")

using Test

eps = 1e-7

@testset "SMSE()" begin
    data = [Dato([1], 1), Dato([1],1), Dato([2], 3)]
    n = RegNode([1, 2, 3])
    setprediction!(n, data)
    @test (SMSE(n,data) - 8/3) < eps
end

