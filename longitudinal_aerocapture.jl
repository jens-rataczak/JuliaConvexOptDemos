using Convex
import MathOptInterface as MOI 
using ECOS
# using SCS
# using Mosek, MosekTools
using CairoMakie
using Printf
using LinearAlgebra
using DifferentialEquations
using ForwardDiff

include("aerocapture_helpers.jl")

function dynamics!(x_dot, x, p::Tuple{AbstractControl, LongitudinalAerocaptureModel}, t)

    # longituindal Vinh dynamics with defined bank rate and heat-load dynamics
    # time derivative is with respect to dimensional time t 
    # this function should be used in the calls to DifferentialEquations.jl because it is non-allocating

    control, model = p
    # unpack
    r = x[1]
    v = x[2]
    γ = x[3]
    σ = x[4]

    μ = gravitational_constant(model)
    D = drag_acceleration(x, model)
    L = lift_acceleration(x, model)

    ṙ = v*sin(γ)
    v̇ = -D - μ*sin(γ)/r^2 
    γ̇ = 1/v*(L*cos(σ) + (v^2 - μ/r)*cos(γ)/r)
    σ̇ = get_control(control, t)
    q̇ = sutton_graves_heat_flux(x, model)/1e4/1e3

    x_dot .= (ṙ, v̇, γ̇, σ̇, q̇)
    nothing
end

function lift_acceleration(x, model::AbstractModel)
    r, v, _ = get_rvg(x, model)
    ρ = atmospheric_density(r, model.planet.atm)
    β = ballistic_coefficient(model)
    LD = lift_to_drag_ratio(model)
    0.5*ρ*v^2*LD/β
end

function drag_acceleration(x, model::AbstractModel)
    r, v, _ = get_rvg(x, model)
    ρ = atmospheric_density(r, model.planet.atm)
    β = ballistic_coefficient(model)
    0.5*ρ*v^2/β
end

function density_gradient(r, model)
    return ForwardDiff.derivative(_r -> atmospheric_density(_r, model.planet.atm), r)
end

function getAB!(A, B::AbstractVector, x, model::LongitudinalAerocaptureModel)
    # assumes density is a function of altitude only
    
    # these are dimensional
    r = x[1]
    v = x[2]
    γ = x[3]
    σ = x[4]
    μ = gravitational_constant(model)
    LD = lift_to_drag_ratio(model)
    β = ballistic_coefficient(model)
    K = sutton_graves_constant(model)
    Rn = nose_radius(model)
    
    ρ = atmospheric_density(r, model.planet.atm)
    dρ_dr = density_gradient(r, model)
    
    # radius derivatives
    A[1, 2] = sin(γ)
    A[1, 3] = v*cos(γ)

    # velocity derivatives
    A[2, 1] = -0.5 * dρ_dr * v^2 / β + 2 * μ * sin(γ) / r^3
    A[2, 2] = -v * ρ / β 
    A[2, 3] = -μ*cos(γ)/r^2

    # fpa derivatives
    A[3, 1] = (2 * μ - r * v^2) * cos(γ) / r^3 / v + 0.5 * dρ_dr * LD * v * cos(σ) / β 
    A[3, 2] = (r + μ / v^2) * cos(γ) / r^2 + 0.5 * ρ * LD * cos(σ) / β 
    A[3, 3] = (μ - r * v^2) * sin(γ) / r^2 / v
    A[3, 4] = -0.5*ρ*LD*v*sin(σ)/β

    # heat load derivatives
    A[5, 1] = 0.5 * K * Rn^(-0.5) * ρ^(-0.5) * dρ_dr * v^3 / 1e4 / 1e3
    A[5, 2] = 3 * K * sqrt(ρ/Rn) * v^2 / 1e4 / 1e3
    
    B[4] = 1.0
end

function discrete_dynamics!(x_dot, x, p::Tuple{Any, Any, Float64, Int64, AbstractModel, NTuple{15, Any}}, tau)
    control, tau_ref, tf, k, model, (A, B, STM, STM_inv, ẋ, Ȧ, Bṁ, Bṗ, Ṡ, ż, aux1, aux2, aux3, aux4, aux5) = p
    n = length(model.x0)
    n1 = n + n^2

    x_ref = @view x[1:n]
    u_ref = get_control(control, tau*tf)

    getAB!(A, B, x_ref, model)
    A .*= tf
    B .*= tf

    # state dynamics
    dynamics!(aux1, x_ref, (control, model), tau*tf)
    ẋ .= aux1 .* tf

    # STM
    STM .= reshape((@view x[n+1:n1]), (n,n))
    STM_inv .= inv(STM)
    mul!(Ȧ, A, STM)

    # B minus matrix
    lambda_minus = (tau_ref[k+1]-tau)/(tau_ref[k+1]-tau_ref[k])
    aux2 .= STM_inv .* lambda_minus
    mul!(Bṁ, aux2, B)

    # B plus matrix
    lambda_plus = (tau - tau_ref[k])/(tau_ref[k+1]-tau_ref[k])
    aux3 .= STM_inv .* lambda_plus
    mul!(Bṗ, aux3, B)

    # S matrix
    mul!(Ṡ, STM_inv, aux1)
    # dynamics!(Ṡ, x_ref, (control, model), tau*tf)

    # z vector 
    B .*= u_ref
    mul!(aux4, A, x_ref)
    aux5 .= aux4 .+ B
    mul!(ż, STM_inv, aux5)
    ż .*= -1.0

    x_dot .= [ẋ; reduce(vcat, Ȧ); Bṁ; Bṗ; Ṡ; ż]

    nothing
end

function apoapsis_jacobian!(Hf, x, model::LongitudinalAerocaptureModel)
    r, v, γ = get_rvg(x, model)
    μ = gravitational_constant(model)

    e = μ/r - v^2/2
    e_target = μ/model.ra_target - r^2*v^2*cos(γ)^2/(2*model.ra_target^2)

    Hf[1] = -μ/r^2 + r*v^2*cos(γ)^2/model.ra_target^2
    Hf[2] = -v + r^2*v*cos(γ)^2/model.ra_target^2
    Hf[3] = -r^2*v^2*cos(γ)*sin(γ)/model.ra_target^2
    return (e-e_target)
end

# function to generate reference trajectory
function generate_ref_traj(t0, x0, t_ref, control, model::AbstractModel)
    tspan = (t0, t_ref[end])
    sol = solve(ODEProblem(dynamics!, x0, tspan, (control, model)), Tsit5(), tstops=t_ref, saveat=t_ref,
                     abstol=1e-12, reltol=1e-12)
    return mapreduce(permutedims, vcat, sol.u)', sol.t
end

function discrete_loop!(N::Int64, Ak, Bmk, Bpk, Sk, zk, t_ref, x_ref, control, model, params, p0, Ak_matrix)
    # dimension of state
    n = length(initial_condition(model))

    # get non-dimensional reference time
    τ_ref = t_ref./t_ref[end]

    # x_discrete = zeros(n, N)
    # x_discrete[:,1] = x_ref[:,1]
    # u_ref = control_knots(control)
    # tf_ref = t_ref[end]
    tspan = (τ_ref[1], τ_ref[2])
    p0[1:n] .= @view x_ref[:,1] 
    prob = ODEProblem(discrete_dynamics!, p0, tspan, (control, τ_ref, t_ref[end], 1, model, params))
    sol = solve_discrete_time_step(prob)
    update_matrices((@view Ak[:,1]), (@view Bmk[:,1]), (@view Bpk[:,1]), (@view Sk[:,1]), (@view zk[:,1]), (@view sol[:,end]), n, Ak_matrix)
    # x_discrete[:,2] = Ak_matrix*x_ref[:,1] + Bmk[:,1]*u_ref[1] + Bpk[:,1]*u_ref[2] + Sk[:,1]*tf_ref + zk[:,1]
    for k = 2:N-1
        p0[1:n] .= @view x_ref[:,k] 
        tspan = (τ_ref[k], τ_ref[k+1])
        sol = solve_discrete_time_step(remake(prob; tspan=tspan, u0=p0, p=(control, τ_ref, t_ref[end], k, model, params)))
        update_matrices((@view Ak[:,k]), (@view Bmk[:,k]), (@view Bpk[:,k]), (@view Sk[:,k]), (@view zk[:,k]), (@view sol[:,end]), n, Ak_matrix)
        # x_discrete[:,k+1] = Ak_matrix*x_ref[:,k] + Bmk[:,k]*u_ref[k] + Bpk[:,k]*u_ref[k+1] + Sk[:,k]*tf_ref + zk[:,k]
    end
    # plot_discrete(t_ref, x_ref, x_discrete, model)
end

function solve_discrete_time_step(prob)
    solve(prob, Tsit5(), reltol=1e-4, abstol=1e-4)
end
 
function update_matrices(Ak, Bmk, Bpk, Sk, zk, sol, n, Ak_matrix)
    n1 = n + n^2
    n2 = n1 + n
    n3 = n2 + n
    n4 = n3 + n
    Ak .= @view sol[n+1:n1]
    Ak_matrix .= reshape(Ak, (n,n))
    mul!(Bmk, Ak_matrix, (@view sol[n1+1:n2]) )
    mul!(Bpk, Ak_matrix, (@view sol[n2+1:n3]) )
    mul!(Sk, Ak_matrix, (@view sol[n3+1:n4]) )
    mul!(zk, Ak_matrix, (@view sol[n4+1:end]) )
end

function get_xmin_xmax()
    # for Neptune
    xmin = [-20_000., -100., -deg2rad(0.2), -deg2rad(15), -1]
    xmax = [20_000., 100., deg2rad(0.2), deg2rad(15), 1]
    return xmin, xmax
end

function solve_problem(N, x_ref, u_ref, tf_ref, Ak, Bmk, Bpk, Sk, Hf, lf, model::AbstractModel, settings::Dict, stats)
    n = length(initial_condition(model))

    # define variables
    dx = Variable(n, N)
    du = Variable(N)
    dtf = Variable()
    η = Variable(N)
    ηp = Variable()
    ν = Variable(n, N-1)
    νr = Variable(2)

    # get scaling matrices
    xmin, xmax = get_xmin_xmax()
   
    umin = -deg2rad(settings["du_scale"])
    umax =  deg2rad(settings["du_scale"])
    tf_min = -settings["dtf_scale"]
    tf_max =  settings["dtf_scale"]

    scale_x, _ = scaling_matrices(xmin, xmax)
    scale_u, _ = scaling_matrices(umin, umax)
    scale_tf, _ = scaling_matrices(tf_min, tf_max)

    scale_x_inv = inv(scale_x)

    # introduce affine transformations
    dx_u = scale_x * dx
    du_u = scale_u * du
    dtf_u = scale_tf * dtf

    ################# CONSTRAINTS ##################

    # initial condtion
    constraints = Constraint[dx[:,1] == zeros(n)]

    # dynamics
    for k = 1:N-1
        push!(constraints, (scale_x_inv*dx_u[:,k+1] == scale_x_inv*(reshape(Ak[:,k], (n, n))*dx_u[:,k] 
                + Bmk[:,k]*du_u[k] + Bpk[:,k]*du_u[k+1] + Sk[:,k]*dtf_u) ) )
    end

    # trust regions
    push!(constraints, norm(dtf, 1) <= ηp)
    push!(constraints, ηp <= 1)
    # push!(constraints, norm(du, 2)  <= 2)
    for k = 1:N
        push!(constraints, norm(dx[:,k],2) + abs(du[k]) <= η[k])
        push!(constraints, norm(u_ref[k] + du_u[k], 1)/deg2rad(20) <= 1)
        push!(constraints, norm(du[k], 1) <= 1)
        # push!(constraints, η[k] <= 2)
        # for i = 1:n 
        #     push!(constraints, norm(dx[i,k]) <= 2)
        # end
    end

    # time limits
    push!(constraints, (tf_ref+dtf_u)/settings["tf_max"] <= 1)
    push!(constraints, (tf_ref+dtf_u)/settings["tf_min"] >= 1)

    # final altitude constraint
    h0 = initial_condition(model)[1]
    push!(constraints, (x_ref[1,end]+dx_u[1,end])/h0 == 1)

    # apoapsis constraint
    scale_apoap = 1/target_apoapsis(model)
    push!(constraints, scale_apoap*(dot(Hf,dx_u[:,end]) + lf) == νr[1])


    # heat load constraint 
    heat_load_constraint = (x_ref[end, end] + dx_u[end,end])/settings["max_heat_load"]
    push!(constraints, heat_load_constraint  <= 1 + νr[2])

    ################# COST FUNCTION ##################

    # define cost terms
    η_weight = settings["ptr_xu"]*ones(N)
    ptr_cost = dot(η_weight, η) + settings["ptr_tf"]*abs(ηp)
    u_cost = settings["u_weight"] * norm((u_ref + du_u)/settings["u_scale"], 1)
    ν_cost = 10000*norm(vec(ν), 1) + 10000*norm(νr, 1)

    # define total cost and problem
    cost = ptr_cost + u_cost + ν_cost
    problem = minimize(cost, constraints)

    ################# SOLVE ##################
    solver = ECOS.Optimizer
    # solver = MOI.OptimizerWithAttributes(Mosek.Optimizer, "QUIET"=> true )#, "MSK_IPAR_INTPNT_SOLVE_FORM"=>2)
    # solver = SCS.Optimizer
    Convex.solve!(problem, solver; silent = true)

    # update reference
    u_opt = u_ref .+ evaluate(du_u)
    tf_opt = tf_ref .+ evaluate(dtf_u)
    x_opt = x_ref .+ evaluate(dx_u)

    log_cost!(stats, (evaluate(ptr_cost), evaluate(u_cost), evaluate(ν_cost)))

    variables = Dict(
        "dx" => evaluate(dx),
        "du" => evaluate(du),
        "virtual_control" => evaluate(ν),
        "trust_region" => evaluate(η),
        "virtual_buffer" => evaluate(νr)
    )

    return u_opt, tf_opt, x_opt, problem.optval, evaluate(du), evaluate(dx), evaluate(dtf), variables
end

function SCP(t0, x0, control, model::AbstractModel, stats, params, p0, Ak_matrix, settings::Dict, problem_matrices)
    t_eval = Vector{Float64}()
    u_ref = Vector{Float64}()
    u_opt = Vector{Float64}()
    x_opt = []

    variables_all = []

    t_ref = time_knots(control)
    
    # check max iters
    maxIter = settings["scp_max_iter"]

    # unpack matrices
    Ak, Bmk, Bpk, Sk, zk, Hf = problem_matrices
    tol = 1e-4
    cost_old = 1e12

    for i = 1:maxIter    
        # generate reference
        x_ref, t_ref = generate_ref_traj(t0, x0, t_ref, control, model)

        N = length(t_ref)
        # get discsrete-time matrices
        discrete_loop!(N, Ak, Bmk, Bpk, Sk, zk, t_ref, x_ref, control, model, params, p0, Ak_matrix)

        # get terminal jacobians
        lf = apoapsis_jacobian!(Hf, (@view x_ref[:,end]), model)
    
        # save reference
        u_ref = control_knots(control)      
        # setup and solve problem
        u_opt, tf_opt, x_opt, cost, du, dx, dtf, variables = solve_problem(N, x_ref, u_ref, t_ref[end], Ak, Bmk,
                                                 Bpk, Sk, Hf, lf, model, settings, stats)

        print_iter_stats(i, N, cost, x_opt, norm(dx), norm(du), tf_opt, model)
        track_ref!(stats, model, t_ref, x_ref, u_ref, dx, du, dtf)
        push!(variables_all, variables)

        # update control
        t_ref[2:end] .= collect(range(t_ref[2], tf_opt, N-1))
        control = update_control!(control, t_ref, u_opt)

        if norm(dx) < settings["scp_tolerance"] || abs(cost-cost_old) < tol*abs(cost) + 1e-3*tol
            break 
        end
        cost_old = cost
    end
    println("------------")
    return control, x_opt, variables_all
end

function print_iter_stats(i::Int64, N::Int64, cost::Float64, x_opt, dx, du, tf, model)
    # get delta-v for x_opt
    dv = two_burn_delta_v((@view x_opt[:,end]), model)
    oop_dv = length(initial_condition(model)) > 5 ? inclination_change((@view x_opt[:,end]), model) : 0.0

    if mod(i, 10) == 0 || i == 1
        @printf("%-5s %-10s %-10s %-10s %-8s %-8s %-8s %-8s\n", "Iter", "cost", "dx", "du", "tf", "dv", "oop_dv", "N")
        @printf("%-5d %-10.3e %-10.3e %-10.3e %-8.1f %-8.2f %-8.2f %-5d\n", i, cost, dx, du, tf, dv, oop_dv, N)
    else
        @printf("%-5d %-10.3e %-10.3e %-10.3e %-8.1f %-8.2f %-8.2f %-5d\n", i, cost, dx, du, tf, dv, oop_dv, N)
    end
end

function scaling_matrices(xmin, xmax)
    make_diagonal(x::Number) = x
    make_diagonal(x) = Diagonal(x)

    offset(x::Number) = 0
    offset(x) = zeros(length(x))

    c = offset(xmin)
    S = make_diagonal(abs.((xmax-xmin)/2))
    return S, c
end

function initialize_scp_params(n, N)
    # initialize model
    A = zeros(n,n)
    B = zeros(n)
    STM = zeros(n,n)
    STM_inv = zeros(n,n)
    ẋ = zeros(n)
    Ȧ = zeros(n,n)
    Bṁ = zeros(n)
    Bṗ = zeros(n)
    Ṡ = zeros(n)
    ż = zeros(n)
    aux1 = zeros(n)
    aux2 = zeros(n,n)
    aux3 = zeros(n,n)
    aux4 = zeros(n)
    aux5 = zeros(n)
    p0 = [zeros(n); reduce(vcat, I(n)); zeros(n); zeros(n); zeros(n); zeros(n)]
    params = (A, B, STM, STM_inv, ẋ, Ȧ, Bṁ, Bṗ, Ṡ, ż, aux1, aux2, aux3, aux4, aux5)
    Ak_matrix = zeros(n,n)

    Ak = zeros(Float64, n^2, N-1)
    Bmk = zeros(Float64, n, N-1)
    Bpk = zeros(Float64, n, N-1)
    Sk = zeros(Float64, n, N-1)
    zk = zeros(Float64, n, N-1)
    Hf = zeros(n)
    problem_matrices = (Ak, Bmk, Bpk, Sk, zk, Hf)

    return params, p0, Ak_matrix, problem_matrices
end

function run(settings::Dict)
    saveResults = false
    saveName = "test_inclin.jld2"

    tf_guess = 520

    model = NeptuneLongitudinalModel()

    t_ref = collect(range(0, 520, settings["N"]))
    u_ref = zeros(settings["N"])
    control = FirstOrderHold(t_ref, u_ref)

    n = length(initial_condition(model))
    stats = initialize_scp_stats()
    params, p0, Ak_matrix, problem_matrices = initialize_scp_params(n, settings["N"])
    optimal_control, optimal_trajectory, vars = SCP(0.0, initial_condition(model), control, model, stats, params, p0, Ak_matrix, settings, problem_matrices)


    # print_stats(t, x, u, settings, model)

    if saveResults
        println("Saving to $saveName")
        jldsave(saveName; t, x, u, settings, model)
    end

    return optimal_control, optimal_trajectory, stats, model, vars
end


settings = Dict(
        "N" => 50,
        "u_weight" => 5e-3,
        "dtf_weight" => 1e-6,
        "ptr_xu" => 5e-3,
        "ptr_tf" => 5e-2,
        "scp_tolerance" => 1e-3,
        "scp_max_iter" => 50,
        "du_scale" => 0.1,
        "u_scale" => deg2rad(1.0),
        "dtf_scale" => 10.,
        "du_trust" => 2.0,
        "tf_max" => 1000.,
        "tf_min" => 200.,
        "max_heat_load" => 100
    )
;

# u, x = run(settings)