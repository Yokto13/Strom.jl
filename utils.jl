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


println([1,2,3])
println(typeof([1,2,3]))
println(mean([1,2,3], 3))
