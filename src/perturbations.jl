struct StochSimulOptions
    display::Bool
    dr_algo::String
    first_period::Int64
    irf::Int64
    LRE_options::LinearRationalExpectationsOptions
    nar::Int64      
    order::Int64
    periods::Int64
    function StochSimulOptions(options::Dict{String,Any})
        display = true
        dr_algo = "GS"
        first_period = 1
        irf = 40
        LRE_options = LinearRationalExpectationsOptions()
        nar = 5
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
            elseif k == "nar"
                nar = v::Int64
            elseif k == "order"
                order = v::Int64
            elseif k == "periods"
                periods = v::Int64
            end
        end
        new(display, dr_algo, first_period, irf, LRE_options, nar, order, periods)
    end
end

function display_stoch_simul(context::Context, options::StochSimulOptions)
    m = context.models[1]
    endogenous_names = get_endogenous_longname(context.symboltable)
    exogenous_names = get_exogenous_longname(context.symboltable)
    results = context.results.model_results[1]
    LRE_results = results.linearrationalexpectations
    stationary_variables = LRE_results.stationary_variables
    
    ## Solution function
    title = "Coefficients of approximate solution function"
    g1 = LRE_results.g1
    data = Matrix{Any}(undef, m.n_states + m.exogenous_nbr + 1, m.endogenous_nbr + 1)
    data[1, 1] = ""
    # row headers
    for i = 1:m.n_states
        data[i+1, 1] = "ϕ($(endogenous_names[m.i_bkwrd_b[i]]))"
    end
    offset = m.n_states + 1
    for i = 1:m.exogenous_nbr
        data[i+offset, 1] = "$(exogenous_names[i])_t"
    end
    # columns
    for j = 1:m.endogenous_nbr
        # header
        data[1, j+1] = "$(endogenous_names[j])_t"
        # data
        for i = 1:m.n_states + m.exogenous_nbr
            data[i+1, j+1] = g1[j, i]
        end
    end
    # Note: ϕ(x) = x_{t-1} - \bar x
    #    note = string("Note: ϕ(x) = x\U0209C\U0208B\U02081 - ", "\U00305", "x")
    note = string("Note: ϕ(x) = x_{t-1} - steady_state(x)")
    println("\n")
    dynare_table(data, title, note = note)

    ## Moments
    title = "THEORETICAL MOMENTS"
    steadystate = results.trends.endogenous_steady_state
    variance_matrix = LRE_results.endogenous_variance
    variance = diag(variance_matrix)
    std = sqrt.(variance)
    data = Matrix{Any}(undef, m.original_endogenous_nbr + 1, 4)
    data[1, 1] = "VARIABLE"
    # row headers
    for i = 1:m.original_endogenous_nbr
        data[i+1, 1] = "$(endogenous_names[i])"
    end
    # column headers
    data[1, 2] = "MEAN"
    data[1, 3] = "STD. DEV."
    data[1, 4] = "VARIANCE"
    # data
    for i = 1:m.original_endogenous_nbr
        data[i+1, 2] = steadystate[i]
        data[i+1, 3] = std[i]
        data[i+1, 4] = variance[i]
    end
    println("\n")
    dynare_table(data, title)

    ## Variance decomposition
    n = m.original_endogenous_nbr
    VD = LRE_results.variance_decomposition
    title = "VARIANCE DECOMPOSITION (in percent)"
    stationary_nbr = count(stationary_variables[1:m.original_endogenous_nbr])
    data = Matrix{Any}(undef, stationary_nbr + 1, m.exogenous_nbr + 1)
    data[1, 1] = "VARIABLE"
    # row headers
    k = 2
    for i = 1:m.original_endogenous_nbr
        if stationary_variables[i]
            data[k, 1] = "$(endogenous_names[i])"
            k += 1
        end
    end
    # columns
    for j = 1:m.exogenous_nbr
        # header
        data[1, j + 1] = "$(exogenous_names[j])"
        # data
        k = 1
        for i = 1:m.original_endogenous_nbr
            if stationary_variables[i]
                data[k + 1, j + 1] = VD[i, j]
                k += 1
            end
        end
    end
    println("\n")
    dynare_table(data, title)

    ## Correlation
    title = "CORRELATION MATRIX"
    corr = correlation(LRE_results.endogenous_variance)
    data = Matrix{Any}(undef, stationary_nbr + 1, stationary_nbr + 1)
    data[1, 1] = ""
    # row headers
    k = 2
    for i = 1:m.original_endogenous_nbr
        if stationary_variables[i]
            data[k, 1] = "$(endogenous_names[i])"
            k += 1
        end
    end
    # columns
    k1 = 2
    for j = 1:m.original_endogenous_nbr
        if stationary_variables[j] 
            # header
            data[1, k1] = "$(endogenous_names[j])"
            # data
            k2 = 2
            for i = 1:m.original_endogenous_nbr
                if stationary_variables[i]
                    data[k2, k1] = corr[i, j]
                    k2 += 1
                end
            end
            k1 += 1
        end
    end
    println("\n")
    dynare_table(data, title)
    
    ## Autocorrelation
    title = "AUTOCORRELATION COEFFICIENTS"
    ar = [zeros(m.endogenous_nbr) for i in 1:options.nar]
    S1a = zeros(m.n_bkwrd + m.n_both, m.endogenous_nbr)
    S1b = similar(S1a)
    S2  = zeros(m.endogenous_nbr - m.n_bkwrd - m.n_both, m.endogenous_nbr)
    ar = autocorrelation!(ar, LRE_results, S1a, S1b, S2, m.i_bkwrd_b, stationary_variables)
    data = Matrix{Any}(undef, stationary_nbr + 1, options.nar + 1)
    data[1, 1] = ""
    # row headers
    k=2
    for i = 1:m.original_endogenous_nbr
        if stationary_variables[i]
            data[k, 1] = "$(endogenous_names[i])"
            k += 1
        end
    end
    # columns
    for j = 1:options.nar
        # header
        data[1, j + 1] = j
        # data
        k = 1
        for i = 1:m.original_endogenous_nbr
            if stationary_variables[i]
                data[k + 1, j + 1] = ar[j][i]
                k += 1
            end
        end
    end
    println("\n")
    dynare_table(data, title)
end

function make_A_B!(
    A::Matrix{Float64},
    B::Matrix{Float64},
    model::Model,
    results::ModelResults,
)
    vA = view(A, :, model.i_bkwrd_b)
    vA .= results.linearrationalexpectations.g1_1
    B .= results.linearrationalexpectations.g1_2
end

function stoch_simul!(context::Context, field::Dict{String,Any})
    options = StochSimulOptions(field["options"])
    m = context.models[1]
    ncol = m.n_bkwrd + m.n_current + m.n_fwrd + 2 * m.n_both
    tmp_nbr = m.dynamic!.tmp_nbr::Vector{Int64}
    ws = DynamicWs(m.endogenous_nbr, m.exogenous_nbr, ncol, sum(tmp_nbr[1:2]))
    stoch_simul_core!(context, ws, options)
end

function stoch_simul_core!(context::Context, ws::DynamicWs, options::StochSimulOptions)
    model = context.models[1]
    modfileinfo = context.modfileinfo
    results = context.results.model_results[1]
    work = context.work
    #check_parameters(work.params, context.symboltable)
    #check_endogenous(results.trends.endogenous_steady_state)
    compute_stoch_simul!(context, ws, work.params, options; variance_decomposition=true)
    if options.display
        display_stoch_simul(context, options)
    end  
    if (periods = options.periods) > 0
        steadystate = results.trends.endogenous_steady_state
        linear_trend = results.trends.endogenous_linear_trend
        y0 = zeros(model.endogenous_nbr)
        simulresults = Matrix{Float64}(undef, periods + 1, model.endogenous_nbr)
        histval = work.histval
        if modfileinfo.has_histval
            for i in eachindex(skipmissing(view(work.histval, size(work.histval, 1), :)))
                y0[i] = work.histval[end, i]
            end
        else
            if work.model_has_trend[1]
                y0 = steadystate - linear_trend
            else
                y0 = steadystate
            end
        end
        C = cholesky(model.Sigma_e)
        x = vcat(zeros(1, model.exogenous_nbr), randn(periods, model.exogenous_nbr) * C.U)
        A = zeros(model.endogenous_nbr, model.endogenous_nbr)
        B = zeros(model.endogenous_nbr, model.exogenous_nbr)
        make_A_B!(A, B, model, results)
        simul_first_order!(simulresults, y0, x, steadystate, A, B, periods)
        if work.model_has_trend[1]
            simulresults .+= collect(0:periods) * transpose(linear_trend)
        end
        first_period = ExtendedDates.UndatedDate(options.first_period)
        endogenous_names = [Symbol(n) for n in get_endogenous_longname(context.symboltable)]
        data = TimeDataFrame(simulresults, first_period, endogenous_names)
        push!(results.simulations, Simulation("", "stoch_simul", data))
    end
end

function check!(context::Context, field::Dict{String,Any})
    Nothing
end

function compute_stoch_simul!(
    context::Context,
    ws::DynamicWs,
    params::Vector{Float64},
    options::StochSimulOptions;
    variance_decomposition::Bool = false
)
    model = context.models[1]
    results = context.results.model_results[1]
    compute_steady_state!(context)
    endogenous = results.trends.endogenous_steady_state
    exogenous = results.trends.exogenous_steady_state
    fill!(exogenous, 0.0)
    compute_first_order_solution!(
        context,
        endogenous,
        exogenous,
        endogenous,
        params,
        model,
        ws,
        options,
        variance_decomposition
    )
end

function compute_first_order_solution!(
    context::Context,
    endogenous::AbstractVector{Float64},
    exogenous::AbstractVector{Float64},
    steadystate::AbstractVector{Float64},
    params::AbstractVector{Float64},
    model::Model,
    ws::DynamicWs,
    options::StochSimulOptions,
    variance_decomposition::Bool
)

    # abbreviations
    LRE = LinearRationalExpectations
    LREWs = LinearRationalExpectationsWs

    results = context.results.model_results[1]
    LRE_results = results.linearrationalexpectations

    jacobian =
        get_dynamic_jacobian!(ws, params, endogenous, exogenous, steadystate, model, 2)
    algo = options.dr_algo
    wsLRE = LREWs(
        algo,
        model.endogenous_nbr,
        model.exogenous_nbr,
        model.exogenous_deterministic_nbr,
        model.i_fwrd_b,
        model.i_current,
        model.i_bkwrd_b,
        model.i_both,
        model.i_static,
    )
    LRE.remove_static!(jacobian, wsLRE)
    if algo == "GS"
        LRE.get_de!(wsLRE, jacobian)
    else
        LRE.get_abc!(wsLRE, jacobian)
    end
    LRE.first_order_solver!(LRE_results, algo, jacobian, options.LRE_options, wsLRE)
    lre_variance_ws = LRE.VarianceWs(
        model.endogenous_nbr,
        model.n_bkwrd + model.n_both,
        model.exogenous_nbr,
        wsLRE,
    )
    compute_variance!(
        LRE_results,
        model.Sigma_e,
        lre_variance_ws,
    )
    if variance_decomposition
        variance_decomposition!(
            LRE_results,
            model.Sigma_e,
            diag(LRE_results.endogenous_variance),
            zeros(model.exogenous_nbr, model.exogenous_nbr),
            zeros(model.endogenous_nbr, model.endogenous_nbr),
            lre_variance_ws
        )
    end
end

