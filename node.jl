include("utils.jl")

mutable struct Node 
    is_leaf::Bool
    pred::Numeric64
    examples_indices::Vector{Int64}
    split_by::Int64
    value::Any
    left::Node
    right::Node
end
