using OrdinaryDiffEq
using NodesAndModes
using Plots

# polynomial degree
N = 75

# Chebyshev nodes: "good" points for interpolation
x = [-cos(k * pi / N) for k in 0:N] 

# create the nodal differentiation matrix
VDM, dVdx = basis(Line(), N, x) # VDM = V in the notes
D = dVdx / VDM # note that A / B = A * inv(B)

function rhs!(du, u, parameters, t)
    D = parameters.D

    # a hacky way to enforce periodic boundary conditions
    u[1] = u[end] 

    du .= -D * u
end

# u0(x) = sin(pi * x)
u0(x) = exp(-25 * x^2)
u0(x) = abs(x) < 0.5

u = u0.(x)
params = (; D)
tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50))

# interpolation to equispaced plotting nodes
x̃ = LinRange(-1, 1, 100)
Ṽ, _ = basis(Line(), N, x̃) 
Vinterp = Ṽ / VDM

Linf_error = maximum(abs.(Vinterp * sol.u[end] - u0.(x̃)))
plot(x̃, Vinterp * sol.u[end], label = "Pseudospectral solution", marker=:dot)
plot!(x̃, u0.(x̃), label = "Exact solution")
title!("Error = $Linf_error")

# @show Linf_error = maximum(abs.(Vp * sol.u[end] - u0.(xp)))
@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(x̃, Vinterp * sol.u[i])
    plot!(x̃, u0.(x̃))
    plot!(ylims = extrema(u0.(x)) .+ (-0.5, 0.5), leg=false, title="Time = $t")
end