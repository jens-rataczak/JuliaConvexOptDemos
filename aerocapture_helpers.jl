import Base.@kwdef
using Interpolations 

abstract type AbstractModel end

@kwdef struct LongitudinalAerocaptureModel{P, V} <: AbstractModel
    x0::Vector{Float64}
    ra_target::Float64
    rp_target::Float64
    planet::P
    vehicle::V
end

@kwdef struct Planet{T, A}
    Re::T
    μ::T
    Ω::T
    J2::T
    sg_constant::T
    atm::A
end


@kwdef struct Vehicle{T}
    mass::T
    A_ref::T
    Rn::T 
    CL::T
    CD::T
end

function low_LD_MLS_like()
    LD_nom = 0.3
    CD_nom = 1.65
    CL_nom = LD_nom*CD_nom
    β_nom = 110
    A_ref_nom = pi*(5/2)^2 # 5 meter diameter
    mass_nom = β_nom*CD_nom*A_ref_nom
    Rn = 1.25
    Vehicle{Float64}(mass_nom, A_ref_nom, Rn, CL_nom, CD_nom)
end

abstract type AbstractAtmosphere end

@kwdef struct ExponentialAtmosphere{T} <: AbstractAtmosphere
    r0::T
    β::T
    ρ0::T 
end

function NeptuneExponentialAtmosphere()
    ExponentialAtmosphere{Float64}(24764000, 2.0395207688623736e-05, 0.004879990938811426)
end

function NeptuneExponential(; isJ2=false, isRot=false)
    atm = NeptuneExponentialAtmosphere()
    Ω = 1.724256845299676e-05
    J2 = 3411e-06
    Planet{Float64, typeof(atm)}(24764000, 6835099.5e9, isRot*Ω,  isJ2*J2, 6.79e-5, atm)
end

function atmospheric_density(r::Number, atm::ExponentialAtmosphere)
    ρ = atm.ρ0*exp(-atm.β*(r - atm.r0))
    return ρ
end

function NeptuneLongitudinalModel(; isJ2 = false, isRot = false)
    planet = NeptuneExponential(isJ2=isJ2, isRot=isRot)
    vehicle = low_LD_MLS_like()
    # initial spherical state
    h       = 1000*1e3;                     # altitude (m)
    v       = 29*1e3;                       # inertial velocity (m/s)
    γ       = deg2rad(-11.64)               # flight path angle
    σ       = deg2rad(10)                   # bank angle
    Q       = 0.0                           # Heat load
    # target orbit
    ra_target = 100_000*1000 + planet.Re
    rp_target = 3_000*1000 + planet.Re

    x0 = [h+planet.Re, v, γ, σ, Q]
    LongitudinalAerocaptureModel{typeof(planet), typeof(vehicle)}(x0, ra_target, rp_target, planet, vehicle)
end

function target_apoapsis(model::AbstractModel)
    model.ra_target
end

function target_periapsis(model::AbstractModel)
    model.rp_target
end

function gravitational_constant(model::AbstractModel)
    model.planet.μ
end

function equatorial_radius(model::AbstractModel)
    model.planet.Re
end

function initial_condition(model)
    model.x0
end

function get_rvg(x, model::LongitudinalAerocaptureModel)
    r, v, γ = x[1], x[2], x[3]
    return r, v, γ
end

function ballistic_coefficient(model::AbstractModel)
    model.vehicle.mass/(model.vehicle.CD * model.vehicle.A_ref)
end
function lift_to_drag_ratio(model::AbstractModel)
    model.vehicle.CL/model.vehicle.CD
end

function sutton_graves_heat_flux(x, model::AbstractModel)
    r, v, _ = get_rvg(x, model)
    ρ = atmospheric_density(r, model.planet.atm)
    K = sutton_graves_constant(model)
    Rn = nose_radius(model)
    K*sqrt(ρ/Rn)*v^3
end


function nose_radius(model::AbstractModel)
    model.vehicle.Rn
end

function sutton_graves_constant(model::AbstractModel)
    model.planet.sg_constant
end

abstract type AbstractControl end

struct FirstOrderHoldController{T,U,S} <: AbstractControl
    t_ref::T
    u_ref::U
    interpolator::S
end

function FirstOrderHold(t_ref, u_ref)
    interp = linear_interpolation(t_ref, u_ref, extrapolation_bc=Line())
    FirstOrderHoldController{typeof(t_ref), typeof(u_ref), typeof(interp)}(t_ref, u_ref, interp)
end

function get_control(controller::AbstractControl, t)
    controller.interpolator(t)
end

function update_control!(controller::FirstOrderHoldController, t_new, u_new)
    interp = linear_interpolation(t_new, u_new, extrapolation_bc=Line())
    FirstOrderHoldController{typeof(t_new), typeof(u_new), typeof(interp)}(t_new, u_new, interp)
end

function time_knots(controller::AbstractControl)
    controller.t_ref 
end

function control_knots(controller::AbstractControl)
    controller.u_ref
end


struct Tracker{T,U}
    r_ref::T
    v_ref::T
    γ_ref::T
    σ_ref::T
    u_ref::T
    t_ref::T
    δr::T
    δv::T
    δγ::T
    δσ::T
    δu::T
    δtf::U
    δQ::T
    term1::U
    term2::U 
    term3::U 
    term4::U 
    term5::U
    term6::U
end

function initialize_scp_stats()
    Tracker{Vector{Vector{Float64}}, Vector{Float64}}([], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [])
end

function two_burn_delta_v(x, model::AbstractModel)#r, v, gamma, mu, ra_target, rp_target)
    a = semi_major_axis(x, model)
    ra = apoapsis_radius(x, model)
    ra_target, rp_target = target_apoapsis(model), target_periapsis(model)
    μ = gravitational_constant(model)

    # periapsis raise
    Δv1 = sqrt(1/ra - 1/(ra+rp_target)) - sqrt(1/ra - 1/(2*a))

    # apoapsis correction
    Δv2 = sqrt(1/rp_target - 1/(ra_target + rp_target)) - sqrt(1/rp_target - 1/(ra+rp_target))

    Δv = sqrt(2*μ)*(abs(Δv1) + abs(Δv2))
    return Δv
end

function semi_major_axis(x, model::AbstractModel)
    r, v, _ = get_rvg(x, model)
    μ = gravitational_constant(model)
    return μ/(2*μ/r - v^2)
end

function apoapsis_radius(x, model::AbstractModel)
    r, v, γ = get_rvg(x, model)
    a = semi_major_axis(x, model)
    μ = gravitational_constant(model)
    ra = a*(1 + sqrt(1 - v^2 * r^2 * cos(γ)^2/(μ*a)))
    return ra
end

function track_ref!(stats::Tracker, model::LongitudinalAerocaptureModel, t_ref, x_ref, u_ref, dx, du, dtf)
    push!(stats.u_ref, u_ref)
    push!(stats.r_ref, (@view x_ref[1,:]))
    push!(stats.v_ref, (@view x_ref[2,:]))
    push!(stats.γ_ref, (@view x_ref[3,:]))
    push!(stats.σ_ref, (@view x_ref[4,:]))
    push!(stats.t_ref, t_ref)
    push!(stats.δr, (@view dx[1,:]))
    push!(stats.δv, (@view dx[2,:]))
    push!(stats.δγ, (@view dx[3,:]))
    push!(stats.δσ, (@view dx[4,:]))
    push!(stats.δQ, (@view dx[5,:]))
    push!(stats.δu, du)
    push!(stats.δtf, dtf)
end

function log_cost!(t::Tracker, cost_terms::Float64)
    push!(t.term1, cost_terms[1])
end

function log_cost!(t::Tracker, cost_terms::NTuple{2, Float64})
    push!(t.term1, cost_terms[1])
    push!(t.term2, cost_terms[2])
end

function log_cost!(t::Tracker, cost_terms::NTuple{3, Float64})
    push!(t.term1, cost_terms[1])
    push!(t.term2, cost_terms[2])
    push!(t.term3, cost_terms[3])
end

function log_cost!(t::Tracker, cost_terms::NTuple{4, Float64})
    push!(t.term1, cost_terms[1])
    push!(t.term2, cost_terms[2])
    push!(t.term3, cost_terms[3])
    push!(t.term4, cost_terms[4])
end

function log_cost!(t::Tracker, cost_terms::NTuple{5, Float64})
    push!(t.term1, cost_terms[1])
    push!(t.term2, cost_terms[2])
    push!(t.term3, cost_terms[3])
    push!(t.term4, cost_terms[4])
    push!(t.term5, cost_terms[5])
end

function log_cost!(t::Tracker, cost_terms::NTuple{6, Float64})
    push!(t.term1, cost_terms[1])
    push!(t.term2, cost_terms[2])
    push!(t.term3, cost_terms[3])
    push!(t.term4, cost_terms[4])
    push!(t.term5, cost_terms[5])
    push!(t.term6, cost_terms[6])
end

function plot_cost_terms(stats, n_terms)
    n_iter = length(get_cost_term(stats, 1))

    f = Figure()
    ax = Axis(f[1,1], xlabel="Iteration", ylabel="Cost", yscale=log10)

    iters = 1:n_iter
    for i = 1:n_terms
        lines!(ax, iters, get_cost_term(stats, i), label="Term $i")
    end
    axislegend(ax, position=:lb, nbanks=2)
    display(f)
    nothing
end

function get_cost_term(t::Tracker, n)
    if n == 1
        return t.term1 
    elseif n == 2
        return t.term2 
    elseif n == 3
        return t.term3 
    elseif n == 4
        return t.term4
    elseif n == 5
        return t.term5
    elseif n == 6
        return t.term6
    end
end

function plot_traj(t, x, u, model, settings)
    Re = equatorial_radius(model)
    f = Figure()
    ax1 = Axis(f[1,1], ylabel="Altitude (km)")
    ax2 = Axis(f[1,2], ylabel="Bank (deg.)")
    ax3 = Axis(f[2,1], xlabel="Time (s)", ylabel="Heat Load (kJ/cm²)")
    ax4 = Axis(f[2,2], xlabel="Time (s)", ylabel="Bank Rate (deg/s)")
    lines!(ax1, t, (x[1,:] .- Re)./1e3, label=:none)
    lines!(ax2, t, rad2deg.(wrap_angle.(x[end-1,:])), label=:none)
    ylims!(ax2, (-200, 200))
    lines!(ax3, t, x[end, :], label=:none)
    lines!(ax4, t, rad2deg.(u))
    if settings["max_heat_load"] <= 1e3
        hlines!(ax3, settings["max_heat_load"], linestyle=:dash, color=:black, label="Constraint")
    end
    display(f)
    nothing
end

function wrap_angle(angle)
    """
    Wrap an angle between -π and π.

    Parameters:
        angle (Float64): The angle in radians.

    Returns:
        Float64: The wrapped angle.
    """
    wrapped_angle = mod(angle + π, 2π)
    if wrapped_angle <= 0
        wrapped_angle += 2π
    end
    return wrapped_angle - π
end