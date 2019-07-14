using Random
mutable struct Network
    numOfLayers::Int
    weight::Array
    bias::Array
    function Network(layers::Array)
        numOfLayers = length(layers)
        weight = [randn(layers[i], layers[i-1]) for i = 2:length(layers)]
        bias = [randn(layers[i], 1) for i = 2:length(layers)]
        return new(numOfLayers, weight, bias)
    end
end

function forwardFeed(net::Network, inp::Array)
    inp = sigmoid.(inp)
    for i = 1:length(net.weight)
        inp = sigmoid.(net.weight[i]*inp .+ net.bias[i])
    end
    return inp
end

function getGradient(net::Network, data::Matrix, expected::Matrix)
    activations, zs = [sigmoid.(data)], [data]
    for (weightSet, biasSet) in zip(net.weight, net.bias)
        push!(zs, weightSet*activations[length(activations)] .+ biasSet)
        push!(activations, sigmoid.(zs[length(zs)]))
    end
    delta = 2*(activations[length(activations)] .- expected)
    deltaW, deltaB = [], []
    for i = net.numOfLayers:-1:2
        zPrime = sigmoidPrime.(zs[i])
        pushfirst!(deltaB, delta.*zPrime)
        pushfirst!(deltaW, deltaB[1]*Matrix(transpose(activations[i-1])))
        delta = Matrix(transpose(Matrix(transpose(deltaB[1]))*net.weight[i-1]))
    end
    return deltaW, deltaB
end

function learn(net::Network, miniBatch::Array, rate::Float64)
    avgDeltaW = [zeros(size(w)) for w in net.weight]
    avgDeltaB = [zeros(size(b)) for b in net.bias]
    for example in miniBatch
        deltaW, deltaB = getGradient(net, example[1], example[2])
        avgDeltaW = avgDeltaW .+ deltaW
        avgDeltaB = avgDeltaB .+ deltaB
    end
    net.weight = net.weight .- ((rate/length(miniBatch)).*avgDeltaW)
    net.bias = net.bias .- ((rate/length(miniBatch)).*avgDeltaB)
end

function SGD(net::Network, trainingData::Array, validationData::Array, batchSize::Int, rate::Float64, epochs::Int)
    for i = 1:epochs
        shuffle!(trainingData)
        l = length(trainingData)
        miniBatches = [trainingData[j:min(j+batchSize-1, l)] for j=1:batchSize:l]
        for batch in miniBatches
            learn(net, batch, rate)
        end
        println("Epoch ", i, " completed. ", test(net, validationData))
    end
end

function test(net::Network, data::Array)
    correct = 0
    for i = 1:length(data)
        out = forwardFeed(net, data[i][1])
        answer = 1
        for j = 1:length(out)
            if out[j,1] > out[answer, 1]
                answer = j
            end
        end
        if answer-1 == data[i][2]
            correct += 1
        end
    end
    return (correct/length(data))
end

function sigmoid(num)
    return 1.0/(1.0+MathConstants.e^(-num))
end

function sigmoidPrime(num)
    return (sigmoid(num)^2)*MathConstants.e^(-num)
end
