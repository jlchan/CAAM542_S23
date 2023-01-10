using OrdinaryDiffEq
using Plots

N = 100
vertices = LinRange(-1, 1, N+1)

# compute grid spacing and cell centers
h = vertices[2] - vertices[1] 
x = 0.5 * (vertices[1:end-1] + vertices[2:end]) 

# create the initial condition
# u0(x) = sin(pi * x)
u0(x) = exp(-25 * x^2)
# u0(x) = abs(x) < 0.5
u = u0.(x) 

flux(u_l, u_r) = 0.5 * (u_l + u_r) # "central" flux
flux(u_l, u_r) = u_l # "upwind" flux

function rhs!(du, u, parameters, t)
    h = parameters.h
    u_left = [u[end]; u]
    u_right = [u; u[1]]
    f = flux.(u_left, u_right)
    for i = 1:length(du)
        du[i] = -(f[i+1] - f[i]) / h
    end
end

# solve the ODE
tspan = (0, 2.0)
params = (; h)
ode = ODEProblem(rhs!, u, tspan, params)

# Tsit5 is a good "default" ODE solver for our systems. For more details, see
# https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Non-Stiff-Problems
sol = solve(ode, Tsit5(), saveat=LinRange(tspan[1], tspan[2], 100))

plot(x, sol.u[end])
plot!(x, u0.(x))

# # create a movie
# @gif for i = 1:length(sol.u)
#     t = sol.t[i]
#     plot(x, sol.u[i])
#     plot!(x, u0.(x))
#     plot!(ylims = extrema(u0.(x)) .+ (-0.5, 0.5), legend=false, title="Time = $t")
# end