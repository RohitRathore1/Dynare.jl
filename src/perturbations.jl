function display_stoch_simul(context::Context)
    m = context.models[1]
    g1 = context.results.model_results[1].linearrationalexpectations.g1
    title = "Coefficients of approximate solution function"
    endogenous_names = get_endogenous_longname(context.symboltable)
    exogenous_names = get_exogenous_longname(context.symboltable)
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
    for j = 1:m.endogenous_nbr
        data[1, j+1] = "$(endogenous_names[j])_t"
        for i = 1:m.n_states+m.exogenous_nbr
            data[i+1, j+1] = g1[j, i]
        end
    end
    # Note: ϕ(x) = x_{t-1} - \bar x
    #    note = string("Note: ϕ(x) = x\U0209C\U0208B\U02081 - ", "\U00305", "x")
    note = string("Note: ϕ(x) = x_{t-1} - steady_state(x)")
    println("\n")
    dynare_table(data, title, note)
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

struct StochSimulOptions
    display::Bool
    dr_algo::String
    first_period::Int64
    irf::Int64
    LRE_options::LinearRationalExpectationsOptions
    order::Int64
    periods::Int64
    function StochSimulOptions(options::Dict{String,Any})
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
        new(display, dr_algo, first_period, irf, LRE_options, order, periods)
    end
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
    compute_stoch_simul!(context, ws, work.params, options)
    if options.display
        display_stoch_simul(context)
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
    options::StochSimulOptions,
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
        results.endogenous_variance,
        LRE_results,
        model.Sigma_e,
        lre_variance_ws,
    )
end

correlation!(c::AbstractMatrix{T}, v::AbstractMatrix{T}, sd::AbstractVector{T}) where T =
    c .= v ./ (sd .* transpose(sd))


function correlation(v::AbstractMatrix{T}) where T
    @assert issymmetric(v)
    sd = sqrt.(diag(v))
    c = similar(v)
    correlation!(c, v, sd)
end

"""
autocovariance!(aa::Vector{<:AbstractMatrix{T}}, a::AbstractMatrix{T}, v::AbstractMatrix{T}, work1::AbstractMatrix{T}, work2::AbstractMatrix{T},order::Int64)

returns a vector of autocovariance matrices E(y_t y_{t-i}') i = 1,...,i for an vector autoregressive process y_t = Ay_{t-1} + Be_t
"""
function autocovariance!(aa::Vector{<:AbstractMatrix{T}}, a::AbstractMatrix{T}, v::AbstractMatrix{T}, work1::AbstractMatrix{T}, work2::AbstractMatrix{T}, order::Int64) where T <: Real
    copy!(work1, v)
    for i in 1:order
        mul!(work2, a, work1)
        copy!(aa[i], work2)
        tmp = work1
        work1 = work2
        work2 = tmp
    end
end
    
"""
autocovariance!(aa::Vector{<:AbstractVector{T}}, a::AbstractMatrix{T}, v::AbstractMatrix{T}, work1::AbstractMatrix{T}, work2::AbstractMatrix{T},order::Int64)

returns a vector of autocovariance vector with elements E(y_{j,t} y_{j,t-i}') j= 1,...,n and i = 1,...,i for an vector autoregressive process y_t = Ay_{t-1} + Be_t
"""
                       
function autocovariance!(aa::Vector{<:AbstractVector{T}}, a::AbstractMatrix{T}, v::AbstractMatrix{T}, work1::AbstractMatrix{T}, work2::AbstractMatrix{T}, order::Int64) where T <: Real
    copy!(work1, v)
    for i in 1:order
        mul!(work2, a, work1)
        @inbounds for j = 1:size(v, 1)
            aa[i][j] = work2[j, j]
        end
        tmp = work1
        work1 = work2
        work2 = tmp
    end
end
    
                       
