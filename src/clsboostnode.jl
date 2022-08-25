include("boostnode.jl")

mutable struct ClsBoostNode <: BoostNode
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

ClsBoostNode(v) = ClsBoostNode(false, 
                                    0,
                                    v,
                                    -1,
                                    -1,
                                    nothing,
                                    nothing,
                                    1,
                                    1,
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
                             )
