#= Neural Network that always outputs 2 =#

include("./model.jl")
using .Model

function train(feed)
    output = feed([rand()])

    return -abs(2.0 - output[1][1])
end

function main()

    sizes::Array{UInt32} = [1, 5, 5, 1]
    GENERATIONS = 100

    Model.init(train, sizes)

    net::Model.Network = Model.Network()

    for generation in 1:GENERATIONS
        Model.train_generation()

        if generation % 100 == 0
            net = Model.get_best()
            println("Output: ", Model.feedforward(net, [rand()]))
        end
    end

    Model.write_to_file(net, "/tmp/file")
    net = Model.read_from_file("/tmp/file")
    println("Output: ", Model.feedforward(net, [rand()]))
end

main()

