mutable struct Network
    numOfLayers::Int
    weight::Array
    bias::Array
    function Network(layers)
        numOfLayers = length(layers)
        weight = [rand(layers[i], layers[i-1]) for i = 2:length(layers)]
        bias = [rand(layers[i]) for i = 2:length(layers)]
        return new(numOfLayers, weight, bias)
    end
end
function forwardFeed(net::Network, inp::Array)
    for i = 1:length(net.weight)
        inp = net.weight[i]*inp + net.bias[i]
    end
    return inp
end
