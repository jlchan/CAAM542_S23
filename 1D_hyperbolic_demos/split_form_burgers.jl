using OrdinaryDiffEq
using StartUpDG
using Plots

function rhs!(du, u, parameters, t)
    (; rd, md, Dr_weak) = parameters 
    (; Pq, Vq, Vf, Dr, LIFT) = rd
    (; rxJ, nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    flux = @. 1/6 * (uP^2 + uM * uP) * nx - 0.5 * max(abs(uP), abs(uM)) * (uP - uM)

    # f = Projection(u^2)
    f = Pq * ((Vq * u).^2)

    # g = Projection(u * dudx)
    dudx = (rxJ .* (Dr * u)) ./ J
    g = Pq * ((Vq * u) .* (Vq * dudx))    

    volume_terms = (1 / 3) * ((Dr_weak * f) + (J .* g))
    du .= -(volume_terms + LIFT * flux) ./ J
end

N = 3 # polynomial degree
num_elements = 64

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

u0(x) = exp(-25 * x^2)

(; x) = md
u = u0.(x)

Dr_weak = rd.M \ (-rd.Dr' * rd.M)
params = (; rd, md, Dr_weak)
tspan = (0.0, 10)
ode = ODEProblem(rhs!, u, tspan, params)

tol = 1e-4
sol = solve(ode, RK4(), abstol=tol, reltol=tol, 
            saveat=LinRange(tspan[1], tspan[2], 100))
plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)

energy_residual = zeros(length(sol.t))
for i in 1:length(sol.t)
    du = similar(u)
    rhs!(du, sol.u[i], params, sol.t[i])
    energy_residual[i] = sum(sol.u[i] .* (rd.M * (du .* md.J)))
end

# plot u' * M * u
p1 = plot!(p1, [sum(u .* (rd.M * (u .* md.J))) for u in sol.u], label="Energy")
p2 = plot!(p2, energy_residual, label="Energy residual")
plot(p1, p2)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(rd.Vp * x, rd.Vp * sol.u[i])
    plot!(ylims = extrema(u0.(x)) .+ (-1, 1), leg=false, title="Time = $t")
end