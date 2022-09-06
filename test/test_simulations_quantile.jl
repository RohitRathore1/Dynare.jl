using Dynare
using Statistics
using Distributions

# get the solution for example1.mod in context
context = @dynare "models/simulations/example1.mod"

m = context.models[1]
r = context.results.model_results[1]
A = Matrix{Float64}(undef, m.endogenous_nbr, m.endogenous_nbr)
B = Matrix{Float64}(undef, m.endogenous_nbr, m.exogenous_nbr)
Dynare.make_A_B!(A, B, m, r)

function simulate_one_trajectory(T)
    X = zeros(T+1, m.exogenous_nbr)
    Y = zeros(T+1, m.endogenous_nbr)
    nx = context.models[1].exogenous_nbr
    Sigma_e = context.models[1].Sigma_e
    d = MvNormal(zeros(nx), Sigma_e)
    c = r.trends.endogenous_steady_state
    y0 = copy(c)
        for i = 1:T+1
            @views X[i, :] .= rand(d)
        end
    return Dynare.simul_first_order!(Y, y0, X, c, A, B, T)
end

function simulate_several_trajectory(iteration_nbr, T)
    #To generate storage for iteration_nbr matrices X and matrices Y
    Xs = Matrix{Float64}[]
    Ys = Matrix{Float64}[]
    for i in 1:iteration_nbr
        X = zeros(T+1, m.exogenous_nbr)
        Y = zeros(T+1, m.endogenous_nbr)
        nx = context.models[1].exogenous_nbr
        Sigma_e = context.models[1].Sigma_e
        d = MvNormal(zeros(nx), Sigma_e)
        c = r.trends.endogenous_steady_state
        y0 = copy(c)
        for i = 1:T+1
            @views X[i, :] .= rand(d)
        end
        Dynare.simul_first_order!(Y, y0, X, c, A, B, T)
        push!(Xs, X)
        push!(Ys, Y)
    end

    return (Xs, Ys)
end