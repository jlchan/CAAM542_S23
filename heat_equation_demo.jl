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
    sigma_flux = @. 0.5 * (sigmaP - sigmaM) * nx + 100 * (uP - uM)
    du .= (Dr * sigma + LIFT * sigma_flux) ./ J
end

N = 1
num_elements = 32
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements)..., rd)

params = (; rd, md)
(; x) = md
# u = @. exp(-25 * sin(3 * pi * x)^2) 
u = randn(size(x))
tspan = (0.0, .01)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 100), 
            callback=AliveCallback(alive_interval=10))

plot(md.x, sol.u[1], leg=false, ylims=(-.1, .3))

@gif for u in sol.u
    plot(md.x, u, leg=false, ylims=(-1, 3))
end

function build_rhs_matrix(rhs!, u_size)
    u = zeros(u_size)
    du = similar(u)
    A = zeros(length(u), length(u))
    for i in axes(A, 1), j in axes(A, 2)
        u[i] = 1
        rhs!(du, u, params, 0.0)
        A[:, i] .= vec(du)
        u[i] = 0
    end
    return A
end

M = md.J[1, 1] * kron(I(md.num_elements), rd.M)
K = M * build_rhs_matrix(rhs!, size(u))
lambda, W = eigen(K)