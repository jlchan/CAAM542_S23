using OrdinaryDiffEq
using StartUpDG
using Plots

N = 1
num_elements = 80

rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements)..., rd)
md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]

    # enforce periodic BCs using central fluxes
    u_flux = parameters.flux.(uP, uM, nx) - uM .* nx
    du .= -(Dr * u + LIFT * u_flux) ./ J
end

u0(x) = exp(-sin(pi * x)^2)

(; x) = md
u = u0.(x)
flux_upwind(uP, uM, nx) = 0.5 * (uP + uM) * nx - 0.5 * (uP - uM)
flux_central(uP, uM, nx) = 0.5 * (uP + uM) * nx 
sol_central = solve(ODEProblem(rhs!, u, (0.0, 100), (; rd, md, flux=flux_central)), 
                    RK4(), saveat=LinRange(tspan[1], tspan[2], 100))
sol_upwind = solve(ODEProblem(rhs!, u, (0.0, 100), (; rd, md, flux=flux_upwind)), 
                    RK4(), saveat=LinRange(tspan[1], tspan[2], 100))

plot()
for sol in [sol_central, sol_upwind]                    
    error = [maximum(abs.(rd.Vp * sol.u[i] .- u0.(rd.Vp * md.x .- sol.t[i]))) for i in eachindex(sol.u)]
    plot!(sol.t, error)
end
plot!()