using StartUpDG
using OrdinaryDiffEq
using Plots

function rhs!(d2u, du, u, parameters, t) # BR-1 discretization
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
    d2u .= (Dr * sigma + LIFT * sigma_flux) ./ J
end

N = 3
num_elements = 32
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements)..., rd)

params = (; rd, md)
(; x) = md
u = @. exp(-25 * x^2)
du = zeros(size(x))

# estimate dt 
h = minimum(md.J) / 2
dt = 0.5 * h

tspan = (0.0, 2.0)
ode = SecondOrderODEProblem(rhs!, du, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, VelocityVerlet(), dt = dt, 
            saveat=LinRange(tspan[1], tspan[2], 100), 
            callback=AliveCallback(alive_interval=10))

@gif for u in sol.u
    plot(md.x, u.x[1], leg=false, ylims=(-3, 3))
end
