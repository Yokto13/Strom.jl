include("utils.jl")
include("node.jl")

mutable struct RegressionTree
    root::Node
    data::Vector{Dato}

    function set_prediction(n::Node)
        sz = size(n.examples_indices)
        out::Float64 = 0.0
        for j = 1:sz
            i = n.examples_indices[j]
            out += data[i].y
        end
        out /= sz
        n.pred = out
    end

    function stop_dividing(n::Node)
        if size(n.examples) < 3
            true
        end
        false
    end

    function evaluate(left::Node, right::Node)
        MSE(left) + MSE(right)
    end

    function MSE(n::Node)
        sz = size(n.examples_indices)
        res::Float64 = 0.0
        for j = 1:sz
            i = n.examples_indices[j]
            res += (n.pred - data[i])^2
        end
        res /= sz
        res
    end

    function split_node(ftr::Int64, val::Float64, n::Node)
        left = Node()
        right = Node()
        ex_sz = size(n.examples_indices)
        for j = 1:ex_sz
            if data[n.example_indices[j]][ftr] < val
                push!(left.example_indices, n.example_indices[j])
            else
                push!(right.example_indices, n.example_indices[j])
            end
        end
        return (left, right)
    end

    function evaluate_split(ftr::Int64, val::Float64, n::Node)
        left, right = split_node(ftr, val, n)
        return evalueate(left, right)
    end

    function get_extremas(ftr::Int64, ex_sz::Int64, n::Node)
        mi::Float64 = Inf
        ma::Float64 = -Inf
        for j = 1:ex_sz
            ex_i = n.examples_indices[j]
            mi = min(mi, data[ex_i][ftr])
            ma = max(ma, data[ex_i][ftr])
        end
        @assert(mi != Inf)
        @assert(ma != -Inf)
        return (mi, ma)
    end

    function best_split(n::Node)
        ftr_sz = size(data[1])
        ex_sz = size(n.examples_indices)
        best_score = Inf
        best_ftr = -1
        best_val = 0.0
        for ftr = 1:ftr_sz
            mi, ma = get_extremas(ftr, ex_sz, n)
            # Later iterate over more split vals and test different ways to get
            # vals
            split_val = (ma + mi) / 2.0
            score_candidate = evaluate_split(ftr, val, n)
            # Nelze se nějak vyhýbat floatům? Vždyť se tu to v nich topí.
            if score_candidate < best_score
                best_score = score_candidate
                best_ftr = ftr
                best_val = split_val
            end
        end
        @assert(best_ftr != -1)
        return split_node(best_ftr, best_val, n)
    end

    function build(parent::Node)
        if stop_dividing(parent)
            parent.is_leaf = true
            set_prediction(parent)
        else
            parent.left, parent.right = best_split(parent)
            build(parent.left)
            build(parent.right)
        end
    end

    function build_tree()
        root = Node(Vector(1:size(data)))
        build(root)
    end
end

data = [
        Dato([1,2,3], 0)
        Dato([1,2,2], 1)
        Dato([1,0,2], 1)
        Dato([10,2,3], 0)
       ]
#rt = RegressionTree(Node(), data)
