include("node.jl")
include("utils.jl")

using Test

eps = 1e-7

# setprediction!
data = [Dato([1], 1), Dato([1],1), Dato([2], 3)]
n = RegNode([1,3])
setprediction!(n, data)
@test (n.pred - 2) < eps
@test_throws Exception setprediction!(n, [])

# stopdividing
n = ClsNode([1,2,3,4])
@test stopdividing(n, x -> length(x.datainds) < 4) == false
n = ClsNode([1,2])
@test stopdividing(n, x -> length(x.datainds) < 4) == true

# MSE
data = [Dato([1], 1), Dato([1],1), Dato([2], 3)]
n = RegNode([1, 2, 3])
setprediction!(n, data)
@test (MSE(n) - 8/9) < eps

# splitnode
data = [Dato([1], 1), Dato([1],1), Dato([2], 3)]
n = RegNode([1, 2, 3])
l, r = splitnode(1, 2, n, data)
@test length(l.datainds) == 2
@test length(r.datainds) == 1

# getextremas
data = [Dato([-11], 1), Dato([1],1), Dato([2], 3)]
n = ClsNode([1, 2, 3])
@test getextremas(1, n, data) == (-11, 2)

# uniform
@test length(uniform(-1,2.3, 10)) == 10
@test typeof(uniform(-1,2.3, 3)[1]) == Float64
mi, ma = getextremas(uniform(2, 10, 100))
@test mi >= 2
@test ma <= 10

# findsplit!
data = [Dato([1], 1), Dato([1.1],1), Dato([2], 3)]
n = RegNode([1, 2, 3])
findsplit!(n, data, 1000)
println("The best splitval is ", n.splitval)
@test n.ftr == 1
@test n.splitval > 1.1
@test n.splitval < 2

# get_stopcondition
rt = RegTree(data, 2, 5)
cond = get_stopcondition(rt)
n = RegNode([1,3])
n.depth = 4
@test !cond(n)
rt.maxdepth = 3
@test cond(n)
rt.maxdepth = 5
rt.minnode = 4
@test cond(n) 
rt.maxdepth = 5
rt.minnode = 1
@test !cond(n) 
rt.maxdepth = nothing
rt.minnode = nothing
@test !cond(n) 
rt.maxdepth = 1
rt.minnode = nothing
@test cond(n) 
rt.maxdepth = nothing
rt.minnode = 10
@test cond(n) 

# build and predict!
data = [
        Dato([1,2,3], 0),
        Dato([1,2,2], 1),
        Dato([1,0,2], 1),
        Dato([10,2,3], 0),
       ]
rt = RegTree(data)
buildtree!(rt)
for d=data
    p = predict(d, rt)
    println(p)
    @test p == predict(d, rt.root)
    @test p == d.y
end

