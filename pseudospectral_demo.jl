using OrdinaryDiffEq
using NodesAndModes

N = 25

# Chebyshev nodes: "good" points for interpolation
x = [-cos(k * pi / N) for k in 0:N] 

# create differentiation matrix
VDM, dVdx = basis(Line(), N, x)
D = dVdx / VDM

function rhs!(du, u, parameters, t)
    (; D) = parameters

    # a hacky way to enforce periodic boundary conditions
    u[1] = u[end] 

    du .= -D * u
end

u0(x) = exp(-25 * x^2)
u = u0.(x)
params = (; D)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, Tsit5(), dt = 1e-5, saveat=LinRange(tspan[1], tspan[2], 50))

# interpolation to equispaced plotting nodes
xp = LinRange(-1, 1, 100)
Vp = vandermonde(Line(), N, xp) / VDM

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(xp, Vp * sol.u[i])
    plot!(xp, u0.(xp))
    plot!(ylims = extrema(u0.(x)) .+ (-0.5, 0.5), leg=false, title="Time = $t")
end