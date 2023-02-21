using StartUpDG

function rhs_first_deriv!(du, u, parameters, t)
    (; rd, md) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    u_flux = @. 0.5 * (uP - uM) * nx
    du .= (Dr * u + LIFT * u_flux) ./ J
end

function rhs!(du, u, parameters, t)
    (; rd, md) = parameters
    (; Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    uP[md.mapB] .= -uM[md.mapB]
    u_flux = @. 0.5 * (uP - uM) * nx
    dudx = (Dr * u + LIFT * u_flux) ./ J

    uxM = Vf * dudx
    uxP = uxM[mapP]
    ux_flux = @. 0.5 * (uxP - uxM) * nx
    du .= (Dr * dudx + LIFT * ux_flux) ./ J
end

N = 1
num_elements = 16
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(Line(), num_elements)..., rd)
# md = make_periodic(md)

params = (; rd, md)
(; x) = md
u = @. exp(-25 * x^2)
# u = @. 1 + .1 * sin(pi * x)  + .2 * cos(pi * x) - .2 * sin(2 * pi * x) - .3 * cos(2 * pi * x) + .1 * sin(3 * pi * x)
# u = randn(size(md.x))
tspan = (0.0, 10.0)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50), 
            callback=AliveCallback(alive_interval=10))

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

M = kron(Diagonal(md.J[1, :]), rd.M)
K = M * build_rhs_matrix(rhs!, size(md.x))
A = M * build_rhs_matrix(rhs_first_deriv!, size(md.x))

lambda, W = eigen(K)
scatter(md.x, reshape(real.(W[:, 1]), rd.Np, :), leg=false)