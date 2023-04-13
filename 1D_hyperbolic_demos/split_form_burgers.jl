using OrdinaryDiffEq
using StartUpDG
using Plots

function rhs!(du, u, parameters, t)
    (; rd, md, Dr_weak) = parameters 
    (; Pq, Vq, Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    flux = @. 1/6 * (uP^2 + uM * uP) * nx

    # f = Projection(u^2)
    f = Pq * ((Vq * u).^2)

    # g = Projection(u * dudx)
    dudx = (rxJ .* (Dr * u)) ./ J
    g = Pq * ((Vq * u) .* (Vq * dudx))    

    volume_terms = (1 / 3) * ((Dr_weak * f) + (J .* g))
    du .= -(volume_terms + LIFT * flux) ./ J
end

N = 5 # polynomial degree
num_elements = 32

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

u0(x) = exp(-25 * x^2)

(; x) = md
u = u0.(x)

Dr_weak = rd.M \ (-rd.Dr' * rd.M)
params = (; rd, md, Dr_weak)
tspan = (0.0, 20)
ode = ODEProblem(rhs!, u, tspan, params)

sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 100))
plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(rd.Vp * x, rd.Vp * sol.u[i])
    plot!(ylims = extrema(u0.(x)) .+ (-1, 1), leg=false, title="Time = $t")
end