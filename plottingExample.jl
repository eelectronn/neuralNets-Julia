clearconsole()
include("dataLoader.jl")
using Plots, Images
training, testing = loadData()
index = 63
data = training[index][1]
label = training[index][2]
println(label)
data = Matrix(transpose(reshape(data, (28,:))))
img = colorview(Gray, data)
plot(img)
