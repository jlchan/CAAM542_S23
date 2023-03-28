using OrdinaryDiffEq
using StartUpDG
using Plots

N = 3 # polynomial degree
num_elements = 16

rd = RefElemData(Tri(), N)
md = MeshData(uniform_mesh(Tri(), num_elements), rd)
# md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md, map_inflow) = parameters 
    (; Vf, Dr, Ds, LIFT) = rd
    (; rxJ, sxJ, ryJ, syJ, nx, ny, J, Jf, mapP) = md

    uM = Vf * u
    uP = uM[mapP]

    uP[map_inflow] .= -uM[map_inflow]

    # advection part
    u_flux = @. 0.5 * (uP - uM) * (nx + ny) - 0.5 * (uP - uM) * abs(nx + ny)
    dudxJ = rxJ .* (Dr * u) + sxJ .* (Ds * u)
    dudyJ = ryJ .* (Dr * u) + syJ .* (Ds * u)
    du .= -(dudxJ + dudyJ + LIFT * (u_flux .* Jf)) ./ J

    # diffusion part
    uP[md.mapB] .= -uM[md.mapB] # enforce {u} = 0 => u⁺ + u⁻ = 0 => u⁺ = -u⁻
    ux_flux = @. 0.5 * (uP - uM) * nx
    uy_flux = @. 0.5 * (uP - uM) * ny
    sigma_x = (rxJ .* (Dr * u) + sxJ .* (Ds * u) + LIFT * (ux_flux .* Jf)) ./ J
    sigma_y = (ryJ .* (Dr * u) + syJ .* (Ds * u) + LIFT * (uy_flux .* Jf)) ./ J

    sigmaM_x = Vf * sigma_x
    sigmaM_y = Vf * sigma_y
    sigmaP_x = sigmaM_x[mapP]
    sigmaP_y = sigmaM_y[mapP]
    sigma_flux = @. 0.5 * (sigmaP_x - sigmaM_x) * nx + 0.5 * (sigmaP_y - sigmaM_y) * ny + (uP - uM)

    d_sigma_x_dx_J = rxJ .* (Dr * sigma_x) + sxJ .* (Ds * sigma_x)
    d_sigma_y_dy_J = ryJ .* (Dr * sigma_y) + syJ .* (Ds * sigma_y)
    div_sigma = d_sigma_x_dx_J + d_sigma_y_dy_J
    du .+= epsilon * (div_sigma + LIFT * (sigma_flux .* Jf)) ./ J

    # add forcing f(x, t) = 1.0
    du .+= 1.0
end

# impose a discontinuous initial condition
(; x, y) = md
# u = u0.(x, y)

u = zeros(rd.Np, md.num_elements)

epsilon = 0.01
map_inflow = findall(@. abs(md.xf + 1) < 1e-12)

params = (; rd, md, epsilon, map_inflow)
tspan = (0.0, 1.0)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 25), 
            callback=AliveCallback(alive_interval=10))

scatter(vec(rd.Vp * x), vec(rd.Vp * y), zcolor=vec(rd.Vp * sol.u[end]), 
        markersize=2, markerstrokewidth=0, legend=false)

# xp, yp = vec(rd.Vp * x), vec(rd.Vp * y)
# @gif for i in eachindex(sol.u)
#     global xp, yp
#     t = sol.t[i]
#     scatter(xp, yp, zcolor=vec(rd.Vp * sol.u[i]), 
#         markersize=2, markerstrokewidth=0, legend=false)
# end