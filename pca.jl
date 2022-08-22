""" 
Functions in this feel were written for author's enjoynment. 
They are not optimized and their tests are very basic.
Most of them are implemented in LinearAlgebra package and this 
package should be used in real applications instead of this.

--arklez
"""
using LinearAlgebra

function eig_vals(A, iters=100)
    for i=1:iters
        Q, R = qr(A)
        A = R * Q
    end
    return A[diagind(A)]
end

function IPM(A, μ, iters, shift=1e-7)
    D = size(A)[1]
    b = ones(D)
    μ += shift
    inverse = inv(A - μ * I(D))
    C = norm(inverse * b)
    for i=1:iters
        b = (inverse * b) / C
        C = norm(inverse * b)
    end
    return b 
end

function eig_vectors(A, λs, iters=50)
    result = []
    for λ=λs
        push!(result, IPM(A, λ, iters))
    end
    return result
end

function get_2nd_base(A, σs, vs)
    result = zeros(length(vs), length(vs[1]))
    for i=1:length(vs)
        start = 1 + (i - 1) * length(vs[1])
        result[start:start + length(vs[1]) - 1] = A * vs[i] / σs[i]
    end
    return result
end


function SVD(A)
    M = transpose(A) * A
    λs = eig_vals(M)
    sort!(λs, rev=true)
    σs = sqrt.(λs)
    V = eig_vectors(M, λs)
    U = get_2nd_base(A, σs, V)
    Σ = 1.0 * I(length(λs))
    for i=1:length(λs)
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
    println(precomputed)
    return A * precomputed
end
