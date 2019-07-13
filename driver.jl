clearconsole()
include("Network.jl")
net = Network([5, 3, 2])
println(net)
println()
data = [[rand(5,1), rand(2,1)] for i=1:10]
learn(net, data, 0.5)
