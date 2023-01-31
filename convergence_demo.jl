using OrdinaryDiffEq
using StartUpDG
using Plots

function compute_error(N, num_elements)
    println("Computing errors for N=$N, num_elements=$num_elements")
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

    u0(x) = exp(-sin(pi * x)^2)

    (; x) = md
    u = u0.(x)
    params = (; rd, md)
    tspan = (0.0, 1.3)
    ode = ODEProblem(rhs!, u, tspan, params)
    sol = solve(ode, RK4(), abstol=1e-10, reltol=1e-10, saveat=LinRange(tspan[1], tspan[2], 50))
    Linf_error = maximum(abs.(rd.Vp * sol.u[end] - u0.(rd.Vp * x .- sol.t[end])))

    return Linf_error
end

plot()
for N in 1:4
    num_elements = [2^i for i in 2:6]
    h = 2 ./ num_elements
    errors = compute_error.(N, num_elements)
    plot!(h, errors, xaxis=:log, yaxis=:log, marker=:dot, label="N=$N")
    plot!(h, 2 * h.^(N+1), xaxis=:log, yaxis=:log, linestyle=:dash, label="h^$(N+1)")
end
plot!(legend=:bottomright)
