using OrdinaryDiffEq
using StartUpDG
using Plots

# polynomial degree
N = 4
num_elements = 32
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements)..., rd)
md = make_periodic(md)

# strong form
function rhs!(dU, U, parameters, t)
    (; rd, md) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    p, u = U[:, :, 1], U[:, :, 2]

    pM, uM = Vf * p, Vf * u
    pP, uP = pM[mapP], uM[mapP]

    p_flux = @. 0.5 * (uP - uM) * nx - 0.5 * (pP - pM)
    u_flux = @. 0.5 * (pP - pM) * nx - 0.5 * (uP - uM)

    dU[:, :, 1] .= -(Dr * u + LIFT * p_flux) ./ J
    dU[:, :, 2] .= -(Dr * p + LIFT * u_flux) ./ J
end

p0(x) = exp(-25 * (x - 0.25)^2)
u0(x) = 0

(; x) = md
u = [p0.(x) ;;; u0.(x)]
params = (; rd, md)
tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50))

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    U = sol.u[i]
    p, u = U[:,:,1], U[:,:,2]

    plot(vec(rd.Vp * x), vec(rd.Vp * p), label = "p")
    plot!(vec(rd.Vp * x), vec(rd.Vp * u), label = "u")
    plot!(ylims=(-1, 1), legend=:topleft)
end