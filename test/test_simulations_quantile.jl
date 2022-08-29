using Dynare
using Statistics
using Distributions

# get the solution for example1.mod in context
context = @dynare "models/simulations/example1.mod"

m = context.models[1]
r = context.results.model_results[1]
A = Matrix{Float64}(undef, m.endogenous_nbr, m.endogenous_nbr)
B = Matrix{Float64}(undef, m.endogenous_nbr, m.exogenous_nbr)
#Dynare.make_A_B!(A, B, m, r)

T = 100; # horizon of simulation
x = zeros(T + 1, m.exogenous_nbr); # deterministic shocks. Period 1 is for initialization
x[2, 1:2] .= [0.01, 0.01] # e and u equal 0.01 in first simulation period and 0 afterwards
Y = Matrix{Float64}(undef, T+1, m.endogenous_nbr) # endogenous variables y c k a h b
c = r.trends.endogenous_steady_state # steady state of endogenous variables
y0 = copy(c) # endogenous variables are initialized to the steady state
p = x -> Dynare.simul_first_order!(Y, y0, x, c, A, B, T)

function f(context)
    nx = context.models[1].exogenous_nbr
    Sigma_e = context.models[1].Sigma_e
    d = MvNormal(zeros(nx), Sigma_e)
    x = rand(d)
        for i in 1:200 
            quantile(p(x[i]), (0.1, 0.3, 0.5, 07, 0.9))
        end
end