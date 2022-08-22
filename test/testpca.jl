include("../pca.jl")

using Test

function vecand(v)
    prod = true
    for x=v
        prod = prod & x
    end
    return prod
end

@testset "eig_vals()" begin
    A = [1 0; 0 1]
    dp = deepcopy(A)
    @test [1, 1] == eig_vals(A)
    @test dp == A
    A = [1 0 0; 0 1 0; 0 0 1]
    dp = deepcopy(A)
    @test [1, 1, 1] == eig_vals(A)
    @test dp == A
    A = [1 1; 0 1]
    dp = deepcopy(A)
    @test [1, 1] == eig_vals(A)
    @test dp == A
    A = [0 1; 0 1]
    dp = deepcopy(A)
    @test [0, 1] == sort(eig_vals(A))
    @test dp == A
    A = [2 0; 0 1]
    dp = deepcopy(A)
    @test [1, 2] == sort(eig_vals(A))
    @test dp == A
end

@testset "IPM()" begin
    A = [1 0; 0 0]
    dp = deepcopy(A)
    μ = 1.0
    eps = 1e-7
    eps2 = 1e-3
    @test vecand(IPM(A, μ, 50) .- eps - [1, 0] .< [0, 0])
    @test dp == A
    A = [0 0; 0 1]
    dp = deepcopy(A)
    μ = 1.0
    eps = 1e-7
    @test vecand(IPM(A, μ, 50) .- eps - [0, 1] .< [0, 0])
    @test dp == A
    A = [1 1; 1 1]
    dp = deepcopy(A)
    μ = 2.0
    eps = 1e-7
    @test vecand(abs.(IPM(A, μ, 50)) .- eps - [0.707107, 0.707107] .< [0, 0])
    @test dp == A
end

@testset "SVD()" begin
    A = [1 1; 0 1]
    dp = deepcopy(A)
    U, Σ, V = SVD(A)
    eps = 1e-4
    @test vecand(Σ - [1.618 0; 0 0.618] .- eps .< [ 0 0 ; 0 0])
    @test dp == A
end
