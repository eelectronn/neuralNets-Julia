clearconsole()
include("dataLoader.jl")
using Plots, Images
training, testing = loadData()
training_data = training[1]
training_label = training[2]
index = 61
data = training_data[index]
label = training_label[index]
println(label)
data = Matrix(transpose(reshape(data, (28,:))))
img = colorview(Gray, data)
plot(img)
