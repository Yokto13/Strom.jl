include("../src/clsnode.jl")

using Test

eps = 1e-7

@testset "gini()" begin
    data = Data([
            Dato([1,2,3], 3),
            Dato([1,2,2], 1),
            Dato([1,0,2], 2),
            Dato([10,2,3], 2),
        ], 3)
    n1 = GiniNode([1, 2, 3, 4])
    setprediction!(n1, data)
    n2 = GiniNode([1, 3, 4])
    setprediction!(n2, data)
    n3 = GiniNode([3, 4])
    setprediction!(n3, data)
    @test gini(n1, data) > gini(n2, data)
    @test gini(n2, data) > gini(n3, data)
end

@testset "entropy()" begin
    data = Data([
            Dato([1,2,3], 3),
            Dato([1,2,2], 1),
            Dato([1,0,2], 2),
            Dato([10,2,3], 2),
           ], 3)
    n1 = EntropyNode([1, 2, 3, 4])
    setprediction!(n1, data)
    n2 = EntropyNode([1, 3, 4])
    setprediction!(n2, data)
    n3 = EntropyNode([3, 4])
    setprediction!(n3, data)
    @test entropy(n1, data) > entropy(n2, data)
    @test entropy(n2, data) > entropy(n3, data)
end
