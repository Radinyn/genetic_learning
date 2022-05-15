module Model

    include("./network.jl")

    mutable struct MODEL
        NET_SIZES
        AGENTS_PER_GEN
        TOP_PERCENT
        MUTATION_CHANCE
        MUTATION_LIMIT
        TRAIN_FUNCTION
        CURRENT_GENERATION
        CURRENT_BEST
    end

    MODEL() = MODEL(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

    const STATE = Ref{MODEL}()

    function init(
        train_function::Function,
        net_sizes::Array{UInt32},
        agents_per_gen::UInt32 = UInt32(1000),
        top_percent::Float64 = 0.15,
        mutation_chance::Float64 = 0.05,
        mutation_limit::Float64 = 0.1,
    )
        STATE[] = MODEL(
            net_sizes,
            agents_per_gen,
            top_percent,
            mutation_chance,
            mutation_limit,
            train_function,
            nothing,
            nothing
        )
    end

    function train_generation()

        if STATE[].CURRENT_GENERATION === nothing
            STATE[].CURRENT_GENERATION = [Network(STATE[].NET_SIZES) for _ in 1:STATE[].AGENTS_PER_GEN]
        end

        #println(length(STATE[].CURRENT_GENERATION), " ", STATE[].AGENTS_PER_GEN)
        #println([STATE[].CURRENT_GENERATION[i] === STATE[].CURRENT_GENERATION[i+1] ? 1 : 0 for i in 1:(STATE[].AGENTS_PER_GEN-1)])

        n = length(STATE[].CURRENT_GENERATION)
        scores = [(i, -Inf) for i in 1:n]

        # Multithreading
        for i in 1:n
            feed(input) = feedforward(STATE[].CURRENT_GENERATION[i], input)
            scores[i] = (i, STATE[].TRAIN_FUNCTION(feed))
        end

        sort!(scores, by = x->x[2], rev=true)
        top_count = floor(UInt32, STATE[].AGENTS_PER_GEN*STATE[].TOP_PERCENT)

        cross(net1, net2) = crossover(net1, net2, STATE[].MUTATION_CHANCE, STATE[].MUTATION_LIMIT)
        getg(i) = STATE[].CURRENT_GENERATION[i]

        STATE[].CURRENT_BEST = STATE[].CURRENT_GENERATION[scores[1][1]]

        STATE[].CURRENT_GENERATION = [i <= top_count ? getg(j) : cross(getg(rand(1:top_count)), getg(rand(1:top_count))) for (i, (j, _)) in enumerate(scores)]
    end

    function get_best()
        return STATE[].CURRENT_BEST
    end

end