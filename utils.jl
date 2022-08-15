import Base: size, getindex, length

Numeric64 = Union{Float64, Int64}

function mean(v::Vector{<:Numeric64}, sz::Int64)
    out::Float64 = 0.0
    for i = 1:sz
        out += v[i]
    end
    out /= sz
    out
end

struct Dato
    x::Vector{Numeric64}
    y::Numeric64
end

size(d::Dato) = (size(d.x))
length(d::Dato) = (length(d.x))
getindex(d::Dato, i::Integer) = (d.x[i])

println([1,2,3])
println(typeof([1,2,3]))
println(mean([1,2,3], 3))

struct Data
    data::Vector{Dato}
    classcnt::Integer
end

size(d::Data) = (size(d.data))
length(d::Data) = (length(d.data))
getindex(d::Data, i::Integer) = (d.data[i])
