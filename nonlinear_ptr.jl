using CairoMakie
using Convex 
using ECOS
using LinearAlgebra
using OrdinaryDiffEq
using DataInterpolations

include("helpers.jl")

function double_integrator_dynamics!(x_dot, x, p, t)
    u, cd = p
    
    # Unpack state
    v = x[3:4]
    
    # velocity unit vector
    v_mag = norm(v)
    v_hat = v / v_mag
    
    # drag force
    D = 0.5 * cd * v_mag^2 * v_hat
    
    # dynamics
    x_dot[1:2] .= v
    x_dot[3:4] .= -D + u
end

function one_shot_double_integrator_dynamics!(x_dot, x, p, t)
    u, cd = p
    
    # Unpack state
    v = x[3:4]
    
    # velocity unit vector
    v_mag = norm(v)
    v_hat = v / v_mag
    
    # drag force
    D = 0.5 * cd * v_mag^2 * v_hat
    
    # dynamics
    x_dot[1:2] .= v
    x_dot[3:4] .= -D + u(t)
end

function get_AB(t, x_ref, cd)
    # Unpack
    v = x_ref[3:4]
    v1, v2 = v[1], v[2]
    v_mag = norm(v)
    
    # initialze
    A_lin = zeros(4, 4)
    B_lin = zeros(4, 2)
    
    # Compute partials
    dD1_dv1 = 0.5 * cd * (2 * v1^2 + v2^2) / v_mag
    dD1_dv2 = 0.5 * cd * v1 * v2
    dD2_dv1 = 0.5 * cd * (v1^2 + 2 * v2^2) / v_mag
    dD2_dv2 = 0.5 * cd * v1 * v2
    
    # Fill out A matrix
    A_lin[1, 3] = 1.0
    A_lin[2, 4] = 1.0
    A_lin[3, 3] = -dD1_dv1
    A_lin[3, 4] = -dD1_dv2
    A_lin[4, 3] = -dD2_dv1
    A_lin[4, 4] = -dD2_dv2 
    
    # Fill out B matrix
    B_lin[3, 1] = 1.0 
    B_lin[4, 2] = 1.0 
      
    return A_lin, B_lin
end

function stm_derivs(x, p, t)
    # unpack parameters
    cd, u_ref = p

    # Get reference solution
    x_ref = x[1:4]
    
    # Get A, B matrices at reference solution 
    A, B = get_AB(t, x_ref, cd)

    dxdt = zeros(length(x))

    # get dynamics for reference
    x_ref_dot = zeros(4)
    double_integrator_dynamics!(x_ref_dot, x_ref, (u_ref, cd), t)

    # x 
    dxdt[1:4] = x_ref_dot

    # STM
    STM = reshape(x[5:20], 4, 4)
    aux = A * STM
    dxdt[5:20] = vec(aux)
    
    # B matrix
    STM_inv = inv(STM)
    aux = STM_inv * B
    dxdt[21:28] = vec(aux)

    # h Vector
    aux = x_ref_dot - A*x_ref - B*u_ref
    dxdt[29:32] = STM_inv*aux
    
    return dxdt
end

## function to compute discsrete matrices at each t_k 
function get_discrete_matrix(t_ref, x0, u_ref, cd)    
    A, B, h = [], [], []

    N = length(t_ref)

    # initial conditions
    STM0 = I(4)
    B0 = zeros(4, 2)
    h0 = zeros(4)

    # initialize x
    x_ref = zeros(4, N)
    x_ref[:,1] = x0

    # loop over each t_k
    for k = 1:N-1
        x0_aug = vcat(x_ref[:,k], vec(STM0), vec(B0), h0)
        t_span = (t_ref[k], t_ref[k+1])
        prob = ODEProblem(stm_derivs, x0_aug, t_span, (cd, u_ref[:,k]))
        sol = solve(prob, Tsit5())
        x_aug = sol.u[end]
        
        # extract matrices
        x_ref[:,k+1] = x_aug[1:4]
        Ak = reshape(x_aug[5:20], 4, 4)
        Bk = Ak * reshape(x_aug[21:28], 4, 2)
        hk = Ak * x_aug[29:32]

        push!(A, Ak)
        push!(B, Bk)
        push!(h, hk)
    end    
    return A, B, h
end

function solve_subproblem(x_ref, u_ref, x0, xf, N, A, B, h)

    # define our optimization variables
    x = Variable(4, N)
    u = Variable(2, N-1)
    η = Variable(N)
    ν = Variable(4, N)
    ν̃ = Variable(N-1)

    # initial condition
    constraints = Constraint[x[:,1] - x0 == zeros(4)]
    # final condition
    push!(constraints, x[:,end] - xf == zeros(4))

    # dynamics
    for k = 1:N-1
        push!(constraints, x[:,k+1] == A[k]*x[:,k] + B[k]*u[:,k] + h[k] + ν[:,k])
    end

    # path constraint and trust regions
    for k = 1:N-1
        push!(constraints, x[1,k] <= 6 + ν̃[k])

        δx = x[:,k] - x_ref[:,k]
        δu = u[:,k] - u_ref[:,k]
        push!(constraints, sumsquares(δx) + sumsquares(δu) <= η[k])
        push!(constraints, η[k] >= 0)
        push!(constraints, ν̃[k] >= 0)
        # push!(constraints, norm(u[:,k]) <= 0.5)
    end

    # cost function 
    u_cost = 0
    for k = 1:N-1
        u_cost += 10*sumsquares(u[:,k])
    end
    # trust region term
    ptr_cost =  1e-1 * norm(η, 2)
    # slack variables term 
    slack_cost = 1000* norm(ν, 1)
    buffer_cost =  1000 * norm(ν̃, 1)

    cost = u_cost + ptr_cost + slack_cost + buffer_cost

    # define problem 
    problem = minimize(cost, constraints)
    solver = ECOS.Optimizer
    Convex.solve!(problem, solver; silent = true)

    x_opt = evaluate(x)
    u_opt = evaluate(u)
    
    return x_opt, u_opt, evaluate(norm(ν,1))+evaluate(norm(ν̃,1)), evaluate(norm(x-x_ref, 2))
end

function scp_loop(x0, xf, t_ref, u_ref, N, cd)
    x_opt, u_opt = [], []

    x_ref_all, u_ref_all = [], []

    x_ref = initialize_reference(x0, xf, N)
    push!(x_ref_all, x_ref)
    push!(u_ref_all, u_ref)
    for n = 1:10        
        A, B, h = get_discrete_matrix(t_ref, x0, u_ref, cd)

        x_opt, u_opt, residual, dx = solve_determ_problem(x_ref, u_ref, x0, xf, N, A, B, h)
        println("ITERATION $n: res = $(residual), dx = $dx")

        # update control and state references 
        u_ref = u_opt
        x_ref = x_opt
        push!(x_ref_all, x_ref)
        push!(u_ref_all, u_ref)
        if dx <= 1
            break 
        end
    end
    return x_opt, u_opt, x_ref_all, u_ref_all
end

function initialize_reference(x0, xf, N)
    x_ref = zeros(4, N)
    t = collect(range(0, 1, N))
    for (k, tk) in enumerate(t)
        x_ref[:,k] = (1-tk)*x0 + tk*xf 
    end
    return x_ref 
end

function run_main()
    x0 = [1, 8, 2, 0]
    xf = [1, 2, -1, 0]

    tf = 15
    cd = 5e-3

    N = 20

    t_ref = collect(range(0, tf, N))
    u_ref = zeros(2, N-1)

    x_opt, u_opt, x_ref_all, u_ref_all = scp_loop(x0, xf, t_ref, u_ref, N, cd)

    # assemble control 
    controller = ConstantInterpolation(u_opt, t_ref[1:end-1], extrapolate=true)

    t_nonlinear, x_nonlinear = run_nonlinear(x0, tf, cd, controller)

    u_nonlinear = controller(t_nonlinear)
    plot_solution(t_ref, x_opt, u_opt, x0, xf, t_nonlinear, x_nonlinear, u_nonlinear)
    plot_convergence(t_ref, x_ref_all, u_ref_all)

    return t_nonlinear, x_nonlinear
end

