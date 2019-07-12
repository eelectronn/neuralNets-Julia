using MLDatasets
function loadData()
    rawTrainData = MNIST.traintensor(Float64)
    trainingLabel = MNIST.trainlabels()
    trainingData = []
    for i = 1:60000
        push!(trainingData, reshape(rawTrainData[:,:,i], (1,:)))
    end
    rawTestData = MNIST.testtensor(Float64)
    testingLabel = MNIST.testlabels()
    testingData = []
    for i = 1:10000
        push!(testingData, reshape(rawTestData[:,:,i], (1,:)))
    end
    return [trainingData, trainingLabel], [testingData, testingLabel]
end
