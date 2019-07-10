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

function getGradient(net::Network, data::Array, expected::Array)
    expected = Matrix(transpose(expected))
    activations, zs = [sigmoid.(Matrix(transpose(data)))], [sigmoid.(Matrix(transpose(data)))]
    for (weightSet, biasSet) in zip(net.weight, net.bias)
        push!(zs, weightSet*activations[length(activations)]+biasSet)
        push!(activations, sigmoid.(zs[length(zs)]))
    end
    delta = 2*(activations[length(activations)]-expected)
    delta_w, delta_b = [], []
    for i = net.numOfLayers:-1:2
        z_prime = sigmoidPrime.(zs[i])
        pushfirst!(delta_b, delta.*z_prime)
        pushfirst!(delta_w, delta_b[1]*Matrix(transpose(activations[i-1])))
        delta = Matrix(transpose(Matrix(transpose(delta_b[1]))*net.weight[i-1]))
    end
    return delta_w, delta_b
end

function sigmoid(num)
    return 1.0/(1.0+MathConstants.e^(-num))
end

function sigmoidPrime(num)
    return (sigmoid(num)^2)*MathConstants.e^(-num)
end
