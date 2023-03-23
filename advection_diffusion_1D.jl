using StartUpDG
using OrdinaryDiffEq
using Plots

function rhs!(du, u, parameters, t)
    (; rd, md, epsilon) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]

    # advection part
    uP[1] = 0.0
    u_flux = @. 0.5 * (uP - uM) * nx - 0.5 * (uP - uM)    
    du .= -(Dr * u + LIFT * u_flux) ./ J

    # diffusion part
    uP[md.mapB] .= -uM[md.mapB] # enforce {u} = 0 => u⁺ + u⁻ = 0 => u⁺ = -u⁻
    u_flux = @. 0.5 * (uP - uM) * nx
    sigma = (Dr * u + LIFT * u_flux) ./ J

    sigmaM = Vf * sigma
    sigmaP = sigmaM[mapP]
    sigma_flux = @. 0.5 * (sigmaP - sigmaM) * nx + (uP - uM)
    du .+= epsilon * (Dr * sigma + LIFT * sigma_flux) ./ J

    # add forcing f(x, t) = 1.0
    du .+= 1.0
end


N = 3
num_elements = 32
rd = RefElemData(Line(), N)
(VX,), EToV = uniform_mesh(Line(), num_elements)
@. VX = 0.5 * (1 + VX) # transform the domain to [0, 1]
md = MeshData(VX, EToV, rd)

epsilon = 0.1
params = (; rd, md, epsilon)
(; x) = md
u = @. 0 * x
tspan = (0.0, 5.0)
ode = ODEProblem(rhs!, u, tspan, params)

u_exact(x, epsilon) = x - (exp(-(1-x) / epsilon) - exp(-1 / epsilon)) / (1 - exp(-1/epsilon))

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, ROCK4(), saveat=LinRange(tspan[1], tspan[2], 100), 
            callback=AliveCallback(alive_interval=10))

u_ex = u_exact.(rd.Vp * md.x, epsilon)
plot(rd.Vp * md.x, rd.Vp * sol.u[end] - u_ex, leg=false)
