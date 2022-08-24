include("boostnode.jl")

mutable struct RegBoostNode <: BoostNode
    isleaf::Bool
    pred::Numeric64
    datainds::Vector{Int64}
    splitval::Numeric64
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
    id::Integer
    λ::Numeric64
end

RegBoostNode(v::Vector{Int64}) = RegBoostNode(false, -1, v, -1, -1, nothing,
                                              nothing, 1, 1, 0)
RegBoostNode() = RegBoostNode(false, -1, [], -1, -1, nothing, nothing, 1, 1, 0,)
RegBoostNode(v::Vector{Int64}, λ) = RegBoostNode(false, -1, v, -1, -1, nothing,
                                                 nothing, 1, 1, λ)
RegBoostNode(λ) = RegBoostNode(false, -1, [], -1, -1, nothing, nothing, 1, 1, λ)

