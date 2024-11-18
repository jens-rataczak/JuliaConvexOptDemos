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