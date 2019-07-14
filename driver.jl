clearconsole()
using Serialization
include("Network.jl")
include("dataLoader.jl")
training, validation, testing = loadData()
println("dataLoaded")
net = Network([784, 30, 10])
SGD(net, training, validation, 10, 1.0, 20)
