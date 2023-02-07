using OrdinaryDiffEq
using StartUpDG
using Plots

N = 4 # polynomial degree
num_elements = 20

h = 2 / num_elements

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters # rd = parameters.rd, md = parameters.md
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]

    u_flux = @. 0.5 * (uP - uM) * nx - 0.5 * (uP - uM)
    du .= -(Dr * u + LIFT * u_flux) ./ J
end

# u0(x) = sin(pi * x)
# u0(x) = exp(-25 * x^2)
u0(x) = abs(x) < 0.5
# u0(x) = exp(-10 * sin(pi * x)^2) # exact solution: u0(x - t)

(; x) = md
u = u0.(x)
params = (; rd, md)
tspan = (0.0, 2)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, RK4(), dt = .5 * h, adaptive=false, saveat=LinRange(tspan[1], tspan[2], 50))

plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)
# println("For N = $N, num_elements = $num_elements, the number of timesteps taken is $(sol.destats.naccept)")
# @show Linf_error = maximum(abs.(rd.Vp * sol.u[end] - u0.(rd.Vp * x .- sol.t[end])))

# @gif for i in eachindex(sol.u)
#     t = sol.t[i]
#     plot(rd.Vp * x, rd.Vp * sol.u[i])
#     plot!(rd.Vp * x, u0.(rd.Vp * x))
#     plot!(ylims = extrema(u0.(x)) .+ (-0.5, 0.5), leg=false, title="Time = $t")
# end