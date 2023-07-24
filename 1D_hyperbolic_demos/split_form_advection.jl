using OrdinaryDiffEq
using StartUpDG
using Plots

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md, a, dadx, Dr_weak) = parameters 
    (; Pq, Vq, Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    aM = Vf * a    
    uP = uM[mapP]
    flux = @. 0.5 * (aM * uP) * nx 

    # f = Projection(a * u)
    f = Pq * ((Vq * a) .* (Vq * u))

    # g = Projection(a * dudx)
    dudx = (rxJ .* (Dr * u)) ./ J
    g = Pq * ((Vq * a) .* (Vq * dudx))    

    # h = Projection(dadx * u)
    h = Pq * ((Vq * dadx) .* (Vq * u))
    
    volume_terms = 0.5 * ((Dr_weak * f) + (J .* g) + (J .* h))
    du .= -(volume_terms + LIFT * flux) ./ J
end

N = 5 # polynomial degree
num_elements = 32

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd)
md = make_periodic(md)

u0(x) = sin(pi * x)
# u0(x) = exp(-25 * x^2)

(; x, rxJ, J) = md
u = u0.(x)
a = @. exp(-sin(pi * x)^2) 
dadx = (rxJ .* (rd.Dr * a)) ./ J
Dr_weak = rd.M \ (-rd.Dr' * rd.M)

params = (; rd, md, a, dadx, Dr_weak)

tspan = (0.0, 15)
ode = ODEProblem(rhs!, u, tspan, params)

sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 100))

plot(rd.Vp * x, rd.Vp * sol.u[end], leg=false)

# @gif for i in eachindex(sol.u)
#     t = sol.t[i]
#     plot(rd.Vp * x, rd.Vp * sol.u[i])
#     # plot!(rd.Vp * x, u0.(rd.Vp * x))
#     plot!(ylims = extrema(u0.(x)) .+ (-2, 2), leg=false, title="Time = $t")
# end