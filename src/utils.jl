import Base: size, getindex, length, iterate, setindex!

function mean(v::Vector{<:Number}, sz::Integer)
    out = 0.0
    for i = 1:sz
        out += v[i]
    end
    out /= sz
    out
end

struct Dato
    x::Vector{Number}
    y
end

size(d::Dato) = (size(d.x))
length(d::Dato) = (length(d.x))
getindex(d::Dato, i::Integer) = (d.x[i])


mutable struct Data <: AbstractArray{Dato, 1}
    data::Vector{Dato}
    classcnt::Integer
end

size(d::Data) = (size(d.data))
length(d::Data) = (length(d.data))
getindex(d::Data, i::Integer) = (d.data[i])
setindex!(d::Data, v, i::Int) = (d.data[i] = v)
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

"""
    softmax(m)

Standard stable softmax applied on the last dim of `m`.
"""
function softmax(m)
    lastdim = length(size(m))
    ma = maximum(m, dims=lastdim)
    exped = exp.(m .- ma)
    exped / sum(exped, dims=lastdim)
end
