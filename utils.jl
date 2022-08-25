import Base: size, getindex, length, iterate, setindex!

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
    y
end

size(d::Dato) = (size(d.x))
length(d::Dato) = (length(d.x))
getindex(d::Dato, i::Integer) = (d.x[i])

println([1,2,3])
println(typeof([1,2,3]))
println(mean([1,2,3], 3))

mutable struct Data <: AbstractArray{Dato, 1}
    data::Vector{Dato}
    classcnt::Integer
end

size(d::Data) = (size(d.data))
length(d::Data) = (length(d.data))
getindex(d::Data, i::Integer) = (d.data[i])
setindex!(d::Data, v, i::Int) = (d.data[i] = v)
# setindex!(d::Data, v, I::Vararg{Int, 1}) = (d.data[I] = v)
IndexStyle(::Data) = (IndexLinear())

function iterate(d::Data)
    return iterate(d, 0)
end
function iterate(d::Data, state)
    if length(d.data) < state+1
        return nothing
    end
    return (d.data[state + 1], state + 1)
end

function MSE(y_hat, y)
    return (y_hat - y) ^ 2 
end

function MSE_all(data, tree)
    all = 0
    for d=data
        all += MSE(predict(d.x, tree), d.y)
    end
    all
end

function invec(item, v)
    for e=v
        if e == item
            return true
        end
    end
    return false
end

function onehot(t, classcnt)
    out = zeros(classcnt)
    out[t] = 1
    return out
end

function argmax(v)
    mv, mi = -Inf, -1
    for i=1:length(v)
        if mv <= v[i]
            mv = v[i]
            mi = i
        end
    end
    return mi
end

function argmaxx(v)
    out = []
    for r=v
        push!(out, argmax(r))
    end
    return out
end

function accuracy(pairs)
    correctcnt = 0
    for p=pairs
        if p[2] == p[1]
            correctcnt+=1 
        end
    end
    return correctcnt / length(pairs)
end

"""
    uniform(mi, ma, cnt)

Generate `cnt` float values uniformly between `mi` and `ma`.
"""
function uniform(mi, ma, cnt)
    out = rand(cnt)
    @assert(ma >= mi)
    diff = ma - mi
    out = out * diff .+ mi
    return out
end

function softmax(v)
    ma = maximum(v)
    exp.(v .- ma) / sum(exp.(v .- ma))
end
