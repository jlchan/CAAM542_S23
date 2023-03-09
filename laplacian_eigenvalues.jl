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

function build_rhs_matrix(rhs!, u_size, params)
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

function build_rhs_matrix(N::Int, num_elements::Int, rhs!, parameters=(; ))
    rd = RefElemData(Line(), N)
    md = MeshData(uniform_mesh(Line(), num_elements)..., rd)
    A = build_rhs_matrix(rhs!, size(md.x), (; rd, md, parameters...))
    M = kron(Diagonal(md.J[1,:]), rd.M)
    return M * A
end

N = 3
num_elements = 40
[eigvals(build_rhs_matrix(N, num_elements, rhs_IPDG!, (; alpha = alpha))) for alpha in LinRange(0, 10 * num_elements, 10)]