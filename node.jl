include("utils.jl")


mutable struct Node 
    is_leaf::Bool
    pred::Numeric64
    examples_indices::Vector{Int64}
    split_by::Int64
    value::Numeric64
    left::Union{Nothing, Node}
    right::Union{Nothing, Node}
end

Node(v::Vector{Int64}) = Node(false, 0, v, -1, -1, nothing, 
                              nothing)
Node() = Node(false, 0, [], -1, -1, nothing, nothing)
