include("boostnode.jl")

mutable struct ClsBoostNode <: BoostNode
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

ClsBoostNode(v::Vector{Int64}) = ClsBoostNode(false, 
                                              0,
                                              v,
                                              -1,
                                              -1,
                                              nothing,
                                              nothing,
                                              1,
                                              1,
                                              0
                                             )

ClsBoostNode() = ClsBoostNode(false,
                              0,
                              [],
                              -1,
                              -1,
                              nothing,
                              nothing,
                              1,
                              1,
                              0
                             )

ClsBoostNode(v::Vector{Int64}, λ) = ClsBoostNode(false,
                                                 0,
                                                 v,
                                                 -1,
                                                 -1,
                                                 nothing,
                                                 nothing, 
                                                 1, 
                                                 1, 
                                                 λ
                                                )

ClsBoostNode(λ) = ClsBoostNode(false,
                               0,
                               [], 
                               -1,
                               -1,
                               nothing,
                               nothing,
                               1,
                               1,
                               λ
                              )

