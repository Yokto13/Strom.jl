include("node.jl")

abstract type BoostNode <: Node end

"""
    evaluate(BoostRegNode, G, H)

Evaluate node.
"""
function evaluate(n::BoostNode, G, H)
    return similarity(n, G, H)
end

function derivativevals(n::BoostNode, G, H)
    sz = length(n.datainds)
    res::Float64 = 0.0
    Hs, Gs = 0, 0
    for j = 1:sz
        i = n.datainds[j]
        Gs += G[j]
        Hs += H[j]
    end
    return Gs, Hs
"""
    similarity(BoostNode, H, G)

Sum of mean squared errors belonging to `n.datainds`,
"""
function similarity(n::BoostNode, G, H)
    Gs, Hs = derivativevals(n, G, H)
    return Gs ^ 2 / (Hs + n.λ)
end

function calcprediction(n, G, H)
    Gs, Hs = derivativevals(n, G, H)
    return Gs / (Hs + n.λ)
end
