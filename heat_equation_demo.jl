using StartUpDG
using OrdinaryDiffEq
using Plots

function rhs!(du, u, parameters, t)
    (; rd, md) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    uP[md.mapB] .= -uM[md.mapB] # enforce {u} = 0 => u⁺ + u⁻ = 0 => u⁺ = -u⁻

    u_flux = @. 0.5 * (uP - uM) * nx
    sigma = (Dr * u + LIFT * u_flux) ./ J

    sigmaM = Vf * sigma
    sigmaP = sigmaM[mapP]
    sigma_flux = @. 0.5 * (sigmaP - sigmaM) * nx + (uP - uM)
    du .= (Dr * sigma + LIFT * sigma_flux) ./ J
end

function rhs_IPDG!(du, u, parameters, t)
    (; rd, md, alpha) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    uP[md.mapB] .= -uM[md.mapB] # enforce {u} = 0 => u⁺ + u⁻ = 0 => u⁺ = -u⁻

    u_flux = @. 0.5 * (uP + uM) * nx
    dudx = (Dr * u) ./ J
    sigma = dudx + (LIFT * (u_flux - uM .* nx)) ./ J

    dudxM = Vf * dudx
    dudxP = dudxM[mapP]
    sigma_flux = @. 0.5 * (dudxP + dudxM) * nx + alpha * (uP - uM)    
    du .= (Dr * sigma + LIFT * (sigma_flux - (Vf * sigma) .* nx)) ./ J
end

N = 3
num_elements = 8
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements)..., rd)

params = (; rd, md, alpha = (N+1)*(N+1) / (2 * minimum(md.J)))
(; x) = md
u = @. exp(-25 * sin(3 * pi * x)^2) 
tspan = (0.0, 0.1)
ode = ODEProblem(rhs_IPDG!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, ROCK4(), saveat=LinRange(tspan[1], tspan[2], 100), 
            callback=AliveCallback(alive_interval=10))

plot(rd.Vp * md.x, rd.Vp * sol.u[end], leg=false, ylims=(-.1, .5))
