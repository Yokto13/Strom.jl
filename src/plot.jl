using Plots
using GraphPlot
using Graphs
using LayeredLayouts
import Gadfly: cm, SVG, draw


"""
    boostloss(forest, data, outputfile=nothing, args...; kwargs...)

Same as `boostloss!` but the forest is not altered in any way.
"""
function boostloss(forest::BoostForest, data, outputfile=nothing, args...; 
        kwargs...)
    forest = deepcopy(forest)    
    return boostloss!(forest, data, outputfile, args...; kwargs...)
end

"""
    boostloss!(forest, data, outputfile=nothing, args...; kwargs...)

Create line plot of loss as trees are trained, return the plot.

If `outputfile` is specifid save the plot to the file.
"""
function boostloss!(forest::BoostForest, data, outputfile=nothing, args...; 
        kwargs...)
    loss = (x, y) -> (x - y) ^ 2
    if forest.node isa ClsBoostNode
        loss = (p, g) -> - log(p[g])
    end
    losses = buildforest!(forest, evloss=loss)
    p = plot(1:length(losses), losses, args...; kwargs...)
    if !isnothing(outputfile)
        savefig(outputfile)
    end
    return p
end

"""
    boosttraintest!(forest, train, test, outputfile=nothing, args...; kwargs...)

Plot how loss changed for train and test during training.

If `outputfile` is specifid save the plot to the file.
"""
function boosttraintest!(forest, train, test, outputfile=nothing, args...;
    kwargs...)
    forest.data = train
    createtrees!(forest)
    test_l, train_l = [], []
    if forest.node isa RegBoostNode
        updateH!(forest)
        updateG!(forest)
        for tree=forest.trees
            step_buildtrees!(tree, forest, forest.node)
            push!(train_l, evaluate(train, forest, (x, y) -> (x - y)^2))
            push!(test_l, evaluate(test, forest, (x, y) -> (x - y)^2))
        end
    elseif forest.node isa ClsBoostNode
        initpreds!(forest)
        for timestamp=1:forest.treecnt
            step_buildtrees!(timestamp, forest, forest.node)
            push!(train_l, evaluate(train, forest, (p, g) -> - log(p[g])))
            push!(test_l, evaluate(test, forest, (p, g) -> - log(p[g])))
        end
    end
    p = plot(1:length(train_l), train_l, ylabel = "train", args...; kwargs...)
    plot!(1:length(train_l), test_l, ylabel = "test", args...; kwargs...)
    if !isnothing(outputfile)
        savefig(outputfile)
    end
    return p
end

"""
    boosttraintest!(forest, train, test, outputfile=nothing, args...; kwargs...)

Same as `boosttraintest!` but the forest is not altered in any way.
"""
function boosttraintest(forest, train, test, outputfile=nothing, args...;
    kwargs...)
    forest = deepcopy(forest)    
    return boosttraintest!(forest, train, test, outputfile, args...; kwargs...)
end

function treecontour(tree, xrange, yrange, outputfile=nothing, args...; kwargs...)
    f(x, y) = argmax(predict([x, y], tree))
    p = contour(xrange, yrange, f, args...; kwargs...)
    plot(p)
    if !isnothing(outputfile)
        savefig(outputfile)
    end
    return p
end

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
function printtree(tree, outname="tree.svg", width=√200cm, height=10cm)
    G, nodecnt, labels = buildgraph(tree)
    xs, ys, paths = solve_positions(Zarate(), G)
    p = gplot(G, xs, ys, nodelabel=labels)
    img = SVG(outname, width, height)
    draw(img, p)
end
