using OrdinaryDiffEq
using Plots

N = 100
vertices = LinRange(-1, 1, N+1)

# grid spacing
h = vertices[2] - vertices[1] 

# cell centers
x = 0.5 * (vertices[1:end-1] + vertices[2:end]) 

# create initial condition
# u0(x) = sin(pi * x)
u0(x) = exp(-25 * x^2)
u = u0.(x) 

flux(u_l, u_r) = 0.5 * (u_l + u_r)

function rhs!(du, u, parameters, t)
    (; h) = parameters
    u_left = [u[end]; u]
    u_right = [u; u[1]]
    f = flux.(u_left, u_right)
    for i in eachindex(du)
        du[i] = -(f[i+1] - f[i]) / h
    end
end

tspan = (0, 8.0)
params = (; h)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan[1], tspan[2], 50))

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(x, sol.u[i])
    plot!(x, u0.(x))
    plot!(ylims=(-.5, 1.5), leg=false, title="Time = $t")
end