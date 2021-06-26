function display_stoch_simul(context::Context)
    m = context.models[1]
    g1 = context.results.model_results[1].linearrationalexpectations.g1
    title = "Coefficients of approximate solution function"
    endogenous_names = get_endogenous_longname(context.symboltable)
    exogenous_names = get_exogenous_longname(context.symboltable)
    data = Matrix{Any}(undef,
                       m.n_states + m.exogenous_nbr + 1,
                       m.endogenous_nbr + 1)
    data[1, 1] = ""
    # row headers
    for i = 1:m.n_states
        data[i + 1, 1] = "ϕ($(endogenous_names[m.i_bkwrd_b[i]])"
    end
    offset = m.n_states + 1
    for i = 1:m.exogenous_nbr
        data[i + offset, 1] = "$(exogenous_names[i])_t"
    end
    for j =  1:m.endogenous_nbr
        data[1, j + 1] = "$(endogenous_names[j])_t"
        for i = 1:m.n_states + m.exogenous_nbr
            data[i + 1, j + 1] = g1[j, i]
        end
    end        
    # Note: ϕ(x) = x_{t-1} - \bar x
    #    note = string("Note: ϕ(x) = x\U0209C\U0208B\U02081 - ", "\U00305", "x")
    note = string("Note: ϕ(x) = x_{t-1} - steady_state(x)")
    println("\n")
    dynare_table(data, title, note)
end

function make_A_B!(A::Matrix{Float64}, B::Matrix{Float64}, model::Model, results::ModelResults)
    vA = view(A, :, model.i_bkwrd_b)
    vA .= results.linearrationalexpectations.g1_1
    B .= results.linearrationalexpectations.g1_2
end

struct StochSimulOptions
    display::Bool
    dr_algo::String
    first_period::Int64
    irf::Int64
    LRE_options::LinearRationalExpectationsOptions
    order::Int64
    periods::Int64
    function StochSimulOptions(options::Dict{String, Any})
        display = true
        dr_algo = "GS"
        first_period = 1
        irf = 40
        LRE_options = LinearRationalExpectationsOptions()
        order = 1
        periods = 0
        print_results = true
        for (k, v) in pairs(options)
            if k == "noprint"
                display = false
            elseif k == "dr_cycle_reduction" && v::Bool
                dr_algo = "CR"
            elseif k == "first_period"    
                first_period = v::Int64
            elseif k == "irf"
                irf = v::Int64
            elseif k == "order"
                order = v::Int64
            elseif k == "periods"
                periods = v::Int64
            end
        end
        new(display, dr_algo, first_period, irf, LRE_options,
            order, periods)
    end
end

function stoch_simul!(context::Context, field::Dict{String, Any})
    options = StochSimulOptions(field["options"])
    m = context.models[1]
    ncol = m.n_bkwrd + m.n_current + m.n_fwrd + 2*m.n_both
    tmp_nbr = m.dynamic!.tmp_nbr::Vector{Int64}
    ws = PeriodJacobianWs(m.endogenous_nbr,
                          m.exogenous_nbr,
                          ncol,
                          sum(tmp_nbr[1:2]))
    stoch_simul_core!(context, ws, options)
end

function stoch_simul_core!(context::Context, ws::PeriodJacobianWs, options::StochSimulOptions)
    model = context.models[1]
    results = context.results.model_results[1]
    work = context.work
    #check_parameters(work.params, context.symboltable)
    #check_endogenous(results.trends.endogenous_steady_state)
    compute_stoch_simul!(context, ws, work.params, options)
    if options.display
        display_stoch_simul(context)
    end
    compute_variance!(context)
    if (periods = options.periods) > 0
        steadystate = results.trends.endogenous_steady_state
        linear_trend = results.trends.endogenous_linear_trend
        y0 = copy(steadystate)
        simulresults = Matrix{Float64}(undef, periods + 1, model.endogenous_nbr)
        histval = work.histval
        if size(histval, 1) == 0
            # no histval
            if work.model_has_trend[1]
                y0 = steadystate - linear_trend
            else
                y0 = steadystate
            end
        else
            # histval present
            y0 = view(histval, 1, :)
        end
        C = cholesky(model.Sigma_e)
        x = vcat(zeros(1, model.exogenous_nbr),
                 randn(periods, model.exogenous_nbr)*C.U)
        A = zeros(model.endogenous_nbr, model.endogenous_nbr)
        B = zeros(model.endogenous_nbr, model.exogenous_nbr)
        make_A_B!(A, B, model, results)
        simul_first_order!(simulresults, y0, x, steadystate, A, B, periods)
        if work.model_has_trend[1]
            simulresults .+= collect(0:periods) * transpose(linear_trend) 
        end
        first_period = options.first_period
        endogenous_names = [Symbol(n) for n in get_endogenous_longname(context.symboltable)]
        data = TimeDataFrame(simulresults, Periods.Undated, first_period, endogenous_names)
        push!(results.simulations, Simulation("", "stoch_simul", data))
    end
end

function check!(context::Context, field::Dict{String, Any})
end

function compute_stoch_simul!(context::Context, ws::PeriodJacobianWs,
                              params::Vector{Float64}, options::StochSimulOptions)
    model = context.models[1]
    results = context.results.model_results[1]
    compute_steady_state!(context)
    endogenous = results.trends.endogenous_steady_state
    exogenous = results.trends.exogenous_steady_state
    fill!(exogenous, 0.0)
    compute_first_order_solution!(results.linearrationalexpectations,
                                  endogenous, exogenous, endogenous,
                                  params, model, ws, options)
end

function compute_first_order_solution!(
    results::LinearRationalExpectationsResults,
    endogenous::AbstractVector{Float64},
    exogenous::AbstractVector{Float64},
    steadystate::AbstractVector{Float64},
    params::AbstractVector{Float64},
    model::Model, ws::PeriodJacobianWs, options::StochSimulOptions)

    # abbreviations
    LRE = LinearRationalExpectations
    LREWs = LinearRationalExpectationsWs

    get_jacobian!(ws, params, endogenous, exogenous, steadystate,
                  model, 2)
    algo = options.dr_algo
    wsLRE = LREWs(algo,
               model.endogenous_nbr,
               model.exogenous_nbr,
               model.exogenous_deterministic_nbr,
               model.i_fwrd_b,
               model.i_current,
               model.i_bkwrd_b,
               model.i_both,
               model.i_static)
    LRE.remove_static!(ws.jacobian, wsLRE)
    if algo == "GS"
        LRE.get_de!(wsLRE, ws.jacobian)
    else
        LRE.get_abc!(wsLRE, ws.jacobian)
    end
    LRE.first_order_solver!(results,
                            algo,
                            ws.jacobian,
                            options.LRE_options,
                            wsLRE)
end

function compute_variance!(context::Context)
    m = context.models[1]
    results = context.results.model_results[1]
    Σy = results.endogenous_variance
    A = results.linearrationalexpectations.gs1
    B1 = zeros(m.n_states, m.exogenous_nbr)
    g1_1 = results.linearrationalexpectations.g1_1
    g1_2 = results.linearrationalexpectations.g1_2
    vr1 = view(g1_2, m.i_bkwrd_b, :)
    B1 .= vr1
    ws = LyapdWs(m.n_states::Int64)
    Σ = zeros(m.n_states, m.n_states)
    tmp = zeros(m.n_states, m.exogenous_nbr)
    mul!(tmp, B1, m.Sigma_e)
    B = zeros(m.n_states, m.n_states)
    mul!(B, tmp, B1')
    extended_lyapd!(Σ, A, B, ws)
    stationary_variables = results.stationary_variables
    fill!(stationary_variables, true)
    state_stationary_variables = view(stationary_variables, m.i_bkwrd_b)
    nonstate_stationary_variables = view(stationary_variables, m.i_non_states)
    if is_stationary(ws)
        state_stationary_nbr = m.n_states 
        nonstate_stationary_nbr = m.endogenous_nbr - m.n_states
    else
        fill!(Σy, NaN)
        state_stationary_variables .= .!ws.nonstationary_variables
        state_stationary_nbr = count(state_stationary_variables)
        vr3 = view(results.linearrationalexpectations.g1_1, m.i_non_states, :)
        for i = 1:(m.endogenous_nbr - m.n_states)
            for j = 1:m.n_states
                if ws.nonstationary_variables[j] && abs(vr3[i, j]) > 1e-10
                    nonstate_stationary_variables[j] = false
                    break
                end
            end
        end
        nonstate_stationary_nbr = count(nonstate_stationary_variables)
    end
    # state / state
    stationary_nbr = state_stationary_nbr + nonstate_stationary_nbr
    A2 = zeros(nonstate_stationary_nbr, state_stationary_nbr)
    vtmp = view(results.linearrationalexpectations.g1_1, m.i_non_states, :)
    A2 .= view(vtmp, nonstate_stationary_variables, state_stationary_variables)
    vtmp = view(Σy, m.i_bkwrd_b, m.i_bkwrd_b)
    vΣy = view(vtmp, state_stationary_variables, state_stationary_variables)
    vΣ = view(Σ, state_stationary_variables, state_stationary_variables) 
    vΣy .= vΣ
    # endogenous / nonstate
    n_non_states = m.endogenous_nbr - m.n_states
    B2 = zeros(nonstate_stationary_nbr, m.exogenous_nbr)
    vtmp = view(results.linearrationalexpectations.g1_2, m.i_non_states, :)
    vr2 = view(vtmp, nonstate_stationary_variables, :)
    B2 .= vr2
    Σ = zeros(stationary_nbr, nonstate_stationary_nbr)
    vg1 = view(results.linearrationalexpectations.g1_1, stationary_variables, state_stationary_variables)
    tmp1 = zeros(stationary_nbr, state_stationary_nbr)
    mul!(tmp1, vg1, vΣy)
    vg1 = view(results.linearrationalexpectations.g1_1, m.i_non_states, :)
    vg11 = view(vg1, nonstate_stationary_variables, state_stationary_variables)
    mul!(Σ, tmp1, transpose(vg11))
    tmp2 = zeros(stationary_nbr, m.exogenous_nbr)
    vg2 = view(results.linearrationalexpectations.g1_2, stationary_variables, :)
    mul!(tmp2, vg2, m.Sigma_e)
    mul!(Σ, tmp2, transpose(B2), 1.0, 1.0)
    vtmp = view(Σy, :, m.i_non_states) 
    vΣy = view(vtmp, stationary_variables, nonstate_stationary_variables)
    vΣy .= Σ
    # nonstate / state
    vtmp1 = view(Σy, m.i_non_states, m.i_bkwrd_b)
    vΣy1 = view(vtmp1, nonstate_stationary_variables, state_stationary_variables)
    vtmp2 = view(Σy, m.i_bkwrd_b, m.i_non_states)
    vΣy2 = view(vtmp2, state_stationary_variables, nonstate_stationary_variables)
    vΣy1 .= transpose(vΣy2)
end
    
function simul_first_order!(results::AbstractMatrix{Float64}, initial_values::AbstractVector{Float64}, x::AbstractVecOrMat{Float64}, c::AbstractVector{Float64}, A::StridedVecOrMat{Float64}, B::StridedVecOrMat{Float64}, periods::Int64)
    oldthreadnbr = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @Threads.threads for t = 2:periods + 1
        r = view(results, t, :)
        e = view(x, t, :)
        mul!(r, B, e)
    end
    BLAS.set_num_threads(oldthreadnbr)
    r_1 = view(results, 1, :)
    r_1 .= initial_values .- c
    for t = 2:periods + 1
        r = view(results, t, :)
        mul!(r, A, r_1, 1.0, 1.0)
        r_1 .+=  c
        r_1 = r
    end
    r_1 .+= c
end
