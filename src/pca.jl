# Functions in this feel were written for author's enjoynment. 
# They are not optimized and their tests are very basic.
# Most of them are implemented in LinearAlgebra package and this 
# package should be used in real applications instead of this.
# --arklez

using LinearAlgebra

function null(A, l, i, eps=1e-7)
    for k=i:size(A)[1]
        if abs(A[k, l]) >= eps
            return false
        end
    end
    return true
end

function gauss_jordan(A, eps=1e-7)
    i, j = 1, 1
    m, n = size(A)
    while true
        kill = true
        for k=i:m
            for l=j:n
                kill = kill & (abs(A[k, l]) < eps)
            end
        end
        if kill
            return A
        end
        while j <= n && null(A, j, i)
            j += 1
        end
        k = i
        while abs(A[k, j]) < eps
            k += 1
        end
        if k != i
            B = A'
            r = B[:, k]
            B[:, k] = B[:, i]
            B[:, i] = r
            A = B'
        end
        if abs(A[i, j] - 1) >= eps
            for k=j+1:n
                A[i, k] /= A[i, j]
            end
            A[i, k] = 1
        end
        for k=1:m
            if k == i && continue; end
            ratio = A[k, j] / A[i, j]
            for l=j:n
                A[k, l] = A[k, l] - ratio * A[i, l]
            end
        end
        i += 1; j += 1
        end
    return A    
end

function read_solution(U, eps=1e-7)
    m, n = size(U)
    sol = zeros(n)
    for i=m:-1:1
        start = nothing
        for j=1:n
            if isnothing(start) && abs(U[i, j]) >= eps
                start = j
            else 
                if !isnothing(start) && sol[j] == 0.0 && abs(U[i, j]) >= eps
                    sol[j] = 1
                end
            end
        end
        if !isnothing(start)
            tot = 0
            for j=(start + 1):n
                tot += sol[j] * U[i, j]
            end
            sol[start] = - tot / U[i, start]
        end
    end
    return sol
end

function eig_vals(A, iters=1000)
    for i=1:iters
        Q, R = qr(A)
        A = R * Q
    end
    return A[diagind(A)]
end

function IPM(A, μ, iters, shift=1e-7)
    D = size(A)[2]
    b = rand(D)
    b /= norm(b)
    μ += shift
    inverse = inv(A - μ * I(D))
    for i=1:iters
        b = (inverse * b)
        b /= norm(b)
    end
    return b 
end

function eig_vectors(A, λs, iters=10000)
    vectors = Dict()
    for λ=λs
        if λ in keys(vectors)
            M = zeros(length(vectors[λ][1]), length(vectors[λ]) + size(A)[1])
            for i=1:length(vectors[λ])
                M[:, i] = vectors[λ][i]
            end
            diff = A - λ * I(size(A)[1])
            diff = diff'
            for i=length(vectors[λ]) + 1:length(vectors[λ]) + size(A)[1]
                M[:, i] = diff[:, i - length(vectors[λ])]
            end
            M = M'
            U = gauss_jordan(M)
            perp = read_solution(U)
            perp /= norm(perp)
            push!(vectors[λ], perp)
        else
            vectors[λ] = [IPM(A, λ, iters)]
        end
    end
    out = []
    used_λs = Set()
    for λ in λs
        if λ in used_λs && continue; end
        for v in vectors[λ]
            push!(out, v)
        end
        push!(used_λs, λ)
    end
    return out
end

function get_2nd_base(A, σs, vs, eps=1e-7)
    result = zeros(size(A)[1], size(A)[1])
    for i=1:length(σs)
        start = 1 + (i - 1) * size(A)[1]
        if abs(σs[i]) >= eps
            result[start:start + size(A)[1] - 1] = A * vs[i] / σs[i]
        else
            U = gauss_jordan(copy(result'))
            perp = read_solution(U)
            perp /= norm(perp)
            result[start:start + size(A)[1] - 1] = perp
        end
    end
    return result
end


function SVD(A, eps=1e-7)
    M = A' * A
    m, n = size(A)
    λs = eig_vals(M)
    sort!(λs, rev=true)
    σs = sqrt.(λs)
    σs = [σ for σ in σs if abs(σ) >= eps]
    V = eig_vectors(M, λs)
    U = get_2nd_base(A, σs, V)
    Σ = zeros(m, n)
    for i=1:length(σs)
        Σ[i, i] = σs[i] 
    end
    V = mapreduce(permutedims, vcat, V)
    return(U, Σ, V)
end

"""
    compute_PCA(A, dims)

Computes PCA and returns `V` that can be used like ROW_EXAMPLES * V.
"""
function compute_PCA(A, dims)
    elcnt = size(A)[1] * size(A)[2]
    U, Σ, V = SVD(A .- sum(A) / elcnt)    
    range = 1:size(V)[1]
    return V'[:, range .<= dims]
end

"""
    PCA(A[,precomputed, dims])

Principal Component Analysis working on rows of A.

One and only one of `precomputed` and `dims` needs to be set.
"""
function PCA(A, precomputed=nothing, dims=nothing)
    if isnothing(precomputed) && isnothing(dims)
        throw(ArgumentError("One of the optional arguments should be set but 
                            they are both `nothing`!"))
    end
    if !isnothing(precomputed) && !isnothing(dims)
        throw(ArgumentError("Only one of the optional arguments should be set but 
                            they are both set!"))
    end
    if isnothing(precomputed)
        precomputed = compute_PCA(A, dims)
    end
    return A * precomputed
end
