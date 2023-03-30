using OrdinaryDiffEq
using StartUpDG
using Plots

N = 4 # polynomial degree
num_elements = 16

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; a, rd, md) = parameters # rd = parameters.rd, md = parameters.md
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    f = a .* u 
    fM = Vf * f
    fP = fM[mapP]
    flux = @. 0.5 * (fP - fM) * nx 
    
    du .= -(Dr * f + LIFT * (flux)) ./ J
end

# u0(x) = sin(8 * pi * x)
u0(x) = exp(-25 * x^2)
# u0(x) = abs(x) < 0.5
# u0(x) = exp(-10 * sin(pi * x)^2) # exact solution: u0(x - t)

(; x) = md
u = u0.(x)
a = @. exp(-sin(pi * x)^2) #(1 - x^2)^5 + 1 
params = (; rd, md, a)
tspan = (0.0, 10)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 100))

plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(rd.Vp * x, rd.Vp * sol.u[i])
    # plot!(rd.Vp * x, u0.(rd.Vp * x))
    plot!(ylims = extrema(u0.(x)) .+ (-2, 2), leg=false, title="Time = $t")
end