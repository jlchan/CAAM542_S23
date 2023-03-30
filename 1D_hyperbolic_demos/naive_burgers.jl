using OrdinaryDiffEq
using StartUpDG
using Plots

N = 4 # polynomial degree
num_elements = 65

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters # rd = parameters.rd, md = parameters.md
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    f = @. u^2/2
    fM = Vf * f
    fP = fM[mapP]
    uM = Vf * u
    uP = uM[mapP]
    flux = @. 0.5 * (fP - fM) * nx - 0.5 * max(abs(uP), abs(uM)) * (uP - uM)
    
    du .= -(Dr * f + LIFT * (flux)) ./ J
end

# u0(x) = sin(8 * pi * x)
u0(x) = -sin(pi * x)
# u0(x) = exp(-25 * x^2) 

(; x) = md
u = u0.(x)
params = (; rd, md)
tspan = (0.0, 1.5)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 100))

plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(rd.Vp * x, rd.Vp * sol.u[i])
    # plot!(rd.Vp * x, u0.(rd.Vp * x))
    plot!(ylims = extrema(u0.(x)) .+ (-2, 2), leg=false, title="Time = $t")
end