
struct Network
    num_of_layers
    sizes
    weights
    biases
end

function _create_network(sizes::Array{UInt32})
    n = length(sizes)

    biases = [rand(y, 1) for y in sizes[2:n]]
    weights = [rand(sizes[i+1], sizes[i]) for i in 1:(n-1)]
    

    return Network(
        n,
        sizes,
        weights,
        biases
    )
end

Network(sizes::Array{UInt32}) = _create_network(sizes)

sigmoid(x) = 1.0/(1.0+exp(-x))

function feedforward(net::Network, a)
    for (b, w) in zip(net.biases, net.weights)
        a = sigmoid.(w * a) .+ b
    end

    return a
end

# Crossover two networks of the same sizes
function crossover(net1::Network, net2::Network, mutation_chance::Float64, strength_limit::Float64)
    net3 = Network(net1.sizes)

    bias_split = rand(1: sum([length(b) for b in net1.biases]))
    weight_split = rand(1: sum([length(w) for w in net1.weights]))

    count = 0
    for i in 1:length(net1.biases)
        for j in 1:length(net1.biases[i])
            if count < bias_split
                net3.biases[i][j] = net1.biases[i][j]
            else
                net3.biases[i][j] = net2.biases[i][j]
            end
            count += 1
        end
    end

    count = 0
    for i in 1:length(net1.weights)
        for j in 1:length(net1.weights[i])
            if count < weight_split
                net3.weights[i][j] = net1.weights[i][j]
            else
                net3.weights[i][j] = net2.weights[i][j]
            end
            count += 1
        end
    end

    mutate!(net3, mutation_chance, strength_limit)
    return net3
end
    
function mutate_single(x::Float64, strength_limit::Float64)
    x += (rand() % strength_limit) * (-1 * Int32(rand() < 0.5))
    return x 
end

function mutate!(net::Network, chance::Float64, strength_limit::Float64)

    for i in 1:length(net.biases)
        for j in 1:length(net.biases[i])
            if rand() < chance
                net.biases[i][j] = mutate_single(net.biases[i][j], strength_limit)
            end
        end
    end

    for i in 1:length(net.weights)
        for j in 1:length(net.weights[i])
            if rand() < chance
                net.weights[i][j] = mutate_single(net.weights[i][j], strength_limit)
            end
        end
    end
end