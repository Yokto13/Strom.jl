include("../src/pca.jl")

using Test
eps = 1e-7

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
    eps2 = 1e-3
    @test vecand(IPM(A, μ, 50) .- eps - [1, 0] .< [0, 0])
    @test dp == A
    A = [0 0; 0 1]
    dp = deepcopy(A)
    μ = 1.0
    @test vecand(IPM(A, μ, 50) .- eps - [0, 1] .< [0, 0])
    @test dp == A
    A = [1 1; 1 1]
    dp = deepcopy(A)
    μ = 2.0
    @test vecand(abs.(IPM(A, μ, 50)) .- eps - [0.707107, 0.707107] .< [0, 0])
    @test dp == A
end

function test_svdproperties(A)
    U, Σ, V = SVD(A)
    σs = []
    for i=1:size(Σ)[1]
        if i > size(Σ)[2]
            break
        end
        push!(σs, Σ[i, i])
        @test Σ[i, i] + eps >= 0
    end
    @test σs == sort(σs, rev=true)
    @test abs(norm(V[:, 1]) - 1) < eps
    for i=2:size(V)[2]
        @test abs(V[:, i - 1]' * V[:, i]) < eps
    end
    @test abs(norm(U[:, 1]) - 1) < eps
    for i=2:size(U)[2]
        @test abs(U[:, i - 1]' * U[:, i]) < eps
    end
    @test abs((U * Σ * V)[1] - A[1]) < eps 
end

@testset "SVD()" begin
    A = [1 1; 0 1]
    dp = deepcopy(A)
    U, Σ, V = SVD(A)
    eps = 1e-4
    @test vecand(Σ - [1.618 0; 0 0.618] .- eps .< [ 0 0 ; 0 0])
    @test dp == A
    test_svdproperties(A)
    A = [1 0; 0 1]
    test_svdproperties(A)
    A = [1 1; 1 1]
    test_svdproperties(A)
    A = [1 1 2; 3 1 1; 0 0 10]
    test_svdproperties(A)
    A = [1 1 ; 3 1 ; 0 0 ]
    test_svdproperties(A)
    test_svdproperties(A')
    A = [1 1 0; 0 1 1;1 0 1]
    test_svdproperties(A)
    U, Σ, V = SVD(A)
    @test vecand(Σ - [2 0 0; 0 1 0; 0 0 1] .- eps .< [ 0 0 0; 0 0 0; 0 0 0])
    A = [1 2; 2 4;1 2]
    test_svdproperties(A)
    U, Σ, V = SVD(A)
    @test vecand(Σ - [√30 0; 0 0; 0 0] .- eps .< [ 0 0 ; 0 0; 0 0])
end

@testset "PCA()" begin
    A = [1 1; 0 1]
    @test_throws ArgumentError PCA(A)
    @test_throws ArgumentError PCA(A, [0 0; 0 0], 2)
end
