using GraphPlot
using Graphs
using LayeredLayouts
using Gadfly

include("../node.jl")

function countnodes(tree::Tree)
    return countnodes(tree.root)
end

function countnodes(n)
    if isnothing(n)
        return 0
    end
    return 1 + countnodes(n.left) + countnodes(n.right)
end

function dfs!(n::Node, G::AbstractGraph, labels::Vector, cnt)
    parentid = cnt
    labels[parentid] = "ftr: " * string(n.ftr) * " splitval: " * string(n.splitval)
    if !isnothing(n.left)
        add_edge!(G, parentid, cnt + 1)
        cnt = dfs!(n.left, G, labels, cnt + 1)
    end
    if !isnothing(n.right)
        add_edge!(G, parentid, cnt + 1)
        cnt = dfs!(n.right, G, labels, cnt + 1)
    end
    return cnt
end

function buildgraph(tree::Tree)
    nodecnt = countnodes(tree)
    G = DiGraph(nodecnt)
    labels = Array{Union{Nothing, String}}(nothing, nodecnt)
    dfs!(tree.root, G, labels, 1)
    return G, nodecnt, labels
end

"""
    printtree(tree, outname, width, height)
    
Creates a simple SVG graphic of the `tree` and saves it to `outname`.

# Arguments
- `tree`
- `outname`
- `width`: of the resulting image, passed directly to Gadfly's SVG.
- `height`: of the resulting image, passed directly to Gadfly's SVG.
"""
function printtree(tree, outname="tree.svg", width=√200cm, heigh=10cm)
    G, nodecnt, labels = buildgraph(tree)
    xs, ys, paths = solve_positions(Zarate(), G)
    p = gplot(G, xs, ys, nodelabel=labels)
    img = SVG(outname, width, height)
    draw(img, p)
end

d = Data([Dato([1,2,3],3), Dato([3,2,1],2), Dato([3,3,3],2), Dato([2,0,2],1), Dato([1,1,1],2)], 3)
ct = ClsTree(d)
buildtree!(ct)

printtree(ct)
