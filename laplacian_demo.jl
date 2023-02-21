using StartUpDG

function rhs!(du, u, parameters, t)
    (; rd, md) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    uP[md.mapB] .= -uM[md.mapB]
    u_flux = @. 0.5 * (uP - uM) * nx
    sigma = (Dr * u + LIFT * u_flux) ./ J

    sigmaM = Vf * sigma
    sigmaP = sigmaM[mapP]
    sigma_flux = @. 0.5 * (sigmaP - sigmaM) * nx
    du .= (Dr * sigma + LIFT * sigma_flux) ./ J
end

N = 1
num_elements = 16
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements)..., rd)

params = (; rd, md)
(; x) = md
u = @. exp(-25 * x^2)
tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50), 
            callback=AliveCallback(alive_interval=10))

@gif for u in sol.u
    plot(md.x, u, leg=false, ylims=(-1, 3))
end

