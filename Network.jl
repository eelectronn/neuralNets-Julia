mutable struct Network
    numOfLayers::Int
    weight::Array
    bias::Array
    function Network(layers::Array)
        numOfLayers = length(layers)
        weight = [rand(layers[i], layers[i-1]) for i = 2:length(layers)]
        bias = [rand(layers[i], 1) for i = 2:length(layers)]
        return new(numOfLayers, weight, bias)
    end
end

function forwardFeed(net::Network, inp::Array)
    inp = sigmoid.(Matrix(transpose(inp)))
    for i = 1:length(net.weight)
        inp = sigmoid.(net.weight[i]*inp + net.bias[i])
    end
    return Matrix(transpose(inp))
end

function getGradient(net::Network, data::Matrix, expected::Matrix)
    activations, zs = [sigmoid.(data)], [sigmoid.(data)]
    for (weightSet, biasSet) in zip(net.weight, net.bias)
        push!(zs, weightSet*activations[length(activations)]+biasSet)
        push!(activations, sigmoid.(zs[length(zs)]))
    end
    delta = 2*(activations[length(activations)]-expected)
    deltaW, deltaB = [], []
    for i = net.numOfLayers:-1:2
        zPrime = sigmoidPrime.(zs[i])
        pushfirst!(deltaB, delta.*zPrime)
        pushfirst!(deltaW, deltaB[1]*Matrix(transpose(activations[i-1])))
        delta = Matrix(transpose(Matrix(transpose(deltaB[1]))*net.weight[i-1]))
    end
    return deltaW, deltaB
end

function learn(net::Network, miniBatch, rate)
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

function sigmoid(num)
    return 1.0/(1.0+MathConstants.e^(-num))
end

function sigmoidPrime(num)
    return (sigmoid(num)^2)*MathConstants.e^(-num)
end
