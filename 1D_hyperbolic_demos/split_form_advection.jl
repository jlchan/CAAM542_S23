using OrdinaryDiffEq
using StartUpDG
using Plots

# strong form
function rhs!(du, u, parameters, t)
    (; a, rd, md) = parameters # rd = parameters.rd, md = parameters.md
    (; Pq, Vq, Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uq, aq = Vq * u, Vq * a
    aM = Vf * a
    uM = Vf * u
    uP = uM[mapP]
    flux = @. 0.25 * (aM * (uP + uM) + aM * (uP - uM)) * nx 

    volume_terms = 0.5 * (Dr_weak * (uq .* aq) + Pq * (aq .* (Vq * Dr * u)))
    
    du .= -(volume_terms + LIFT * flux) ./ J
end

N = 4 # polynomial degree
num_elements = 16

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

u0(x) = sin(pi * x)
# u0(x) = exp(-25 * x^2)

(; x) = md
u = u0.(x)
a = @. exp(-sin(pi * x)^2) 

Dr_weak = rd.M \ (-rd.Dr' * rd.M * rd.Pq)
params = (; rd, md, a, Dr_weak)
tspan = (0.0, 15)
ode = ODEProblem(rhs!, u, tspan, params)

tol = 1e-3
sol = solve(ode, RK4(), abstol = tol, reltol = tol, saveat=LinRange(tspan[1], tspan[2], 100))

plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(rd.Vp * x, rd.Vp * sol.u[i])
    # plot!(rd.Vp * x, u0.(rd.Vp * x))
    plot!(ylims = extrema(u0.(x)) .+ (-2, 2), leg=false, title="Time = $t")
end