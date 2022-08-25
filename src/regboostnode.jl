include("boostnode.jl")

mutable struct RegBoostNode <: BoostNode
    isleaf::Bool
    pred::Number
    datainds::Vector{Int64}
    splitval::Number
    ftr::Integer
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
    depth::Integer
    id::Integer
end

RegBoostNode(v) = RegBoostNode(false, -1, v, -1, -1, nothing,
                                              nothing, 1, 1)
RegBoostNode() = RegBoostNode(false, -1, [], -1, -1, nothing, nothing, 1, 1)

