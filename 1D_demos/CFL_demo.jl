using StartUpDG
using Plots

function compute_DG_matrix(N, num_elements)

    rd = RefElemData(Line(), N)
    md = MeshData(uniform_mesh(Line(), num_elements)..., rd)
    md = make_periodic(md)

    flux(uP, uM, nx) = 0.5 * (uP + uM) * nx - 0.5 * (uP - uM)

    # strong form
    function rhs!(du, u, parameters, t)
        (; rd, md) = parameters
        (; Vf, Dr, LIFT) = rd
        (; nx, J, mapP) = md

        uM = Vf * u
        uP = uM[mapP]

        # enforce periodic BCs using central fluxes
        u_flux = flux.(uP, uM, nx) - uM .* nx
        du .= -(Dr * u + LIFT * u_flux) ./ J
    end

    params = (; rd, md)

    A = zeros((N+1) * num_elements, (N+1) * num_elements)
    u = zeros(N+1, num_elements)
    du = similar(u)
    for i in 1:(N+1) * num_elements
        u[i] = 1
        rhs!(du, u, params, 0.0)
        A[:,i] .= vec(du)
        u[i] = 0        
    end

    return A
end

A = compute_DG_matrix(3, 8)

spectral_radius(N, num_elements) = 
    maximum(abs.(eigvals(compute_DG_matrix(N, num_elements))))

num_elements = 10:10:80
plot(num_elements, spectral_radius.(3, num_elements), marker=:dot)