using OrdinaryDiffEq
using StartUpDG
using Plots

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters 
    (; Pq, Vq, Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uq, uM = Vq * u, Vf * u
    uP = uM[mapP]
    # flux = @. 1/6 * ((uP^2 + uM^2) + uM * (uP - uM)) * nx - 0.5 * max(abs(uM), abs(uP)) * (uP - uM)
    flux = @. 1/4 * (uP^2 + uM^2) * nx - 1/6 * uM^2 * nx - 0.5 * max(abs(uM), abs(uP)) * (uP - uM)

    volume_terms = 1/3 * (Dr_weak * (uq.^2) + Pq * (uq .* (Vq * Dr * u)))
    
    du .= -(volume_terms + LIFT * flux) ./ J
end

N = 4 # polynomial degree
num_elements = 32

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

u0(x) = 1 + sin(pi * x)
# u0(x) = exp(-25 * x^2)

(; x) = md
u = u0.(x)

Dr_weak = rd.M \ (-rd.Dr' * rd.M * rd.Pq)
params = (; rd, md, Dr_weak)
tspan = (0.0, .5)
ode = ODEProblem(rhs!, u, tspan, params)

# tol = 1e-2
# plot()
# for solver in [Midpoint(), Heun(), SSPRK43(), RK4()]
#     sol = solve(ode, solver,  abstol = tol, reltol = tol, saveat=LinRange(tspan[1], tspan[2], 100))
#     energy(u) = sum(md.wJq .* (rd.Vq * u).^2)
#     plot!(sol.t, energy.(sol.u), label=string(typeof(solver).name.name))
# end
# display(plot!())

tol = 1e-4
sol = solve(ode, RK4(),  abstol = tol, reltol = tol, saveat=LinRange(tspan[1], tspan[2], 100))
plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(rd.Vp * x, rd.Vp * sol.u[i])
    plot!(ylims = extrema(u0.(x)) .+ (-.5, .5), leg=false, title="Time = $t")
end