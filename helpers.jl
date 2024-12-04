function run_nonlinear(x0, tf, cd, controller)
    t_span = (0, tf)
    prob = ODEProblem(one_shot_double_integrator_dynamics!, x0, t_span, (controller, cd))
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    return sol.t, mapreduce(permutedims, vcat, sol.u)'
end

function plot_solution(t, x, u, x0, xf, t_nl, x_nl, u_nl, r1_max)

    f = Figure()
    ax = Axis(f[1,1], xlabel="Time (s)", ylabel="Control")

    lines!(ax, t_nl, u_nl[1,:], label="Integrated")
    lines!(ax, t_nl, u_nl[2,:], label="Integrated")
    scatter!(t[1:end-1], u[1,:], label="SCP Solution")
    scatter!(t[1:end-1], u[2,:], label="SCP Solution")
    axislegend(ax, merge=true, position=:lt)

    f2 = Figure()
    ax2 = Axis(f2[1,1], xlabel = "r₁", ylabel="r₂")
    lines!(x_nl[1,:], x_nl[2,:], label="Integrated")
    scatter!(ax2, x[1,:], x[2,:], label="SCP Solution")
    scatter!(ax2, x0[1], x0[2], color=:black, label="Start")
    scatter!(ax2, xf[1], xf[2], color=:red, label="Target")
    vlines!(ax2, r1_max, color=:black, linestyle=:dash, label="Constraint")
    axislegend(ax2, position=:lc)

    display(f)
    display(f2)
    nothing
end

function plot_convergence(t, x, u)

    f = Figure()
    ax = Axis(f[1,1], xlabel = "r₁", ylabel="r₂")

    cmap = :winter
    for (i, xi) in enumerate(x) 
        lines!(ax, xi[1,:], xi[2,:], label="Iteration $(i-1)", color=i/length(x), colorrange=(0, 1), colormap=Reverse(cmap) )
        scatter!(ax, xi[1,:], xi[2,:], label="Iteration $(i-1)", color=i/length(x), colorrange=(0, 1), colormap=Reverse(cmap) )
    end
    axislegend(ax, merge=true, position=:rb)
    display(f)
    nothing
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

function circle(h, k, r)
    θ = collect(range(0, 2π, 100))
    h .+ r*sin.(θ), k .+ r*cos.(θ)
end

function covariance_ellipse(A; nSigma=3)
    n_points = size(A,2)
    avg = mean(A, dims=2)
    B = A .- avg .* ones(1, n_points)

    U, S, V = svd(B/sqrt(n_points-1))

    θ = (0:.01:1)*2π
    one_sigma = U*diagm(S)*stack([cos.(θ), sin.(θ)], dims=1)
    x_3sigma = avg[1] .+ nSigma .* one_sigma[1,:]
    y_3sigma = avg[2] .+ nSigma .* one_sigma[2,:]
    return x_3sigma, y_3sigma, avg
end

function get_point_stats(x, n_cases, N)
    y = []
    for k = 1:N
        A = stack([x[n][1:2, k] for n = 1:n_cases])
        push!(y, covariance_ellipse(A))
    end
    return y
end

function get_point_std(x, n_cases, N)
    y = zeros(2,N)
    for k = 1:N
        A = stack([x[n][1:2, k] for n = 1:n_cases])
        y[:,k] = std(A, dims = 2)
    end
    return y
end

function run_feedback(x0_aug, control, cd, K, tf, t_ref, noise)
    tspan = (0, tf)
    prob = ODEProblem(double_integrator_feedback_dynamics!, x0_aug, tspan, (control, cd, K, noise))
    sol = solve(prob, Tsit5(), saveat=t_ref, tstops=t_ref, abstol=1e-8, reltol=1e-8)
    return mapreduce(permutedims, vcat, sol.u)', sol.t
end

function run_and_plot(x0, xf, P0, Pf, control, cd, K, x_nom, t_ref, tf, γ, r1_max; n_cases=100)

    x_all, x, u_all, t_all = [], [], [], []
    sampler = MvNormal(x0, P0)
    noise_sampler = MvNormal(zeros(2), γ * I(2))
    K_control = ConstantInterpolation(K, t_ref[1:end-1], extrapolate=true)
    for n = 1:n_cases
        w = [rand(noise_sampler) for k = 1:length(t_ref)]
        noise = DataInterpolations.ConstantInterpolation(w, t_ref, extrapolate=true)
        x0_aug = [x0; rand(sampler)]
        x, t = run_feedback(x0_aug, control, cd, K_control, tf, t_ref, noise)
        u = get_control_history(t, x, K_control, control)
        push!(x_all, x[5:end,:])
        push!(u_all, u)
        push!(t_all, t)
    end
    y = get_point_stats(x_all, n_cases, size(x, 2))
    uy = get_point_stats(u_all, n_cases, size(x, 2))

    f = Figure()
    ax = Axis(f[1,1], aspect=1, xlabel="r₁", ylabel="r₂", title="Trajectory - Full")
   
    for xy in y
        lines!(ax, xy[1], xy[2], color=:red, label="3σ Ellipse")
    end
    lines!(ax, x_nom[1,:], x_nom[2,:], color=:black, label="Mean")
    scatter!(ax, x_nom[1,:], x_nom[2,:], label="Mean")
    vlines!(ax, r1_max, color=:black, linestyle=:dash, label="Chance Constraint")
    lines!(ax, circle(xf[1], xf[2], 3*sqrt(Pf[1,1]))..., color=:black, label="3σ Target Cov")
    lines!(ax, circle(x0[1], x0[2], 3*sqrt(P0[1,1]))..., color=:black, label="3σ Initial Cov")
    axislegend(ax, unique=true, merge=true, position=:lc)
    display(f)

    f2 = Figure()
    ax2 = Axis(f2[1,1],aspect=1, limits=((-1,0.2), (-0.6, 0.6)), xlabel="u₁", ylabel="u₂", title="Control - Full")
    for xy in uy
        lines!(ax2, xy[1], xy[2], color=:red, label="3σ Ellipse")
    end
    u_nom = hcat(control(t_ref))
    u_nom = mapreduce(permutedims, vcat, u_nom)'
    lines!(ax2, u_nom[1,:], u_nom[2,:], color=:black, label="Mean")
    scatter!(ax2, u_nom[1,:], u_nom[2,:], label="Mean")
    scatter!(ax2, u_nom[1,1], u_nom[2,1], color=:green, label="Start")
    axislegend(ax2, unique=true, merge=true, position=:rb)
    display(f2)

    return x_all
end

function get_control_history(t, x, K, u_ref)
    u_all = zeros(2, length(t))
    for k in eachindex(t)
        u_all[:,k] = u_ref(t[k]) + K(t[k])*(x[1:4,k] .- x[5:end,k])
    end
    return u_all
end