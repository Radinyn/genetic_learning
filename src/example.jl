#= Neural Network that always outputs 2 =#

include("./model.jl")
using .Model

function train(feed)
    output = feed([rand()])

    return -abs(2.0 - output[1][1])
end

function main()

    sizes::Array{UInt32} = [1, 5, 5, 1]
    GENERATIONS = 1000

    Model.init(train, sizes)

    for generation in 1:GENERATIONS
        Model.train_generation()

        if generation % 100 == 0
            net::Model.Network = Model.get_best()
            println("Output: ", Model.feedforward(net, [rand()]))
        end
    end
end

main()

