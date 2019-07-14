using MLDatasets
function loadData()
    rawTrainData = MNIST.traintensor(Float64)
    trainingLabel = MNIST.trainlabels()
    trainingData = []
    validationData = []
    for i = 1:50000
        push!(trainingData, [reshape(rawTrainData[:,:,i], (784,:)), vectorizeLabel(trainingLabel[i])])
    end
    for i = 50001:60000
        push!(validationData, [reshape(rawTrainData[:,:,i], (784,:)), trainingLabel[i]])
    end
    rawTestData = MNIST.testtensor(Float64)
    testingLabel = MNIST.testlabels()
    testingData = []
    for i = 1:10000
        push!(testingData, [reshape(rawTestData[:,:,i], (784,:)), testingLabel[i]])
    end
    return trainingData, validationData, testingData
end

function vectorizeLabel(num)
    label = zeros(10,1)
    label[num+1, 1] = 1.0
    return label
end
