using OrdinaryDiffEq
using StartUpDG
using Plots

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md, map_inflow) = parameters 
    (; Vf, Dr, Ds, LIFT) = rd
    (; rxJ, sxJ, ryJ, syJ, nxJ, nyJ, nx, ny, J, Jf, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    u_jump = uP - uM
    
    dudr = Dr * u
    duds = Ds * u

    # advection part
    bx, by = 1, 2 # β vector 
    b_n = bx * nx + by * ny
    uP[map_inflow] .= -uM[map_inflow]
    u_flux = @. 0.5 * u_jump * b_n - 0.5 * abs(b_n) * u_jump
    dudxJ = rxJ .* dudr + sxJ .* duds
    dudyJ = ryJ .* dudr + syJ .* duds
    du .= -(bx * dudxJ + by * dudyJ + LIFT * (u_flux .* Jf)) ./ J

    # diffusion part
    uP[md.mapB] .= -uM[md.mapB] # enforce {u} = 0 => u⁺ + u⁻ = 0 => u⁺ = -u⁻
    @. u_jump = uP - uM
    sigma_x = (dudxJ + LIFT * (0.5 * u_jump .* nxJ)) ./ J
    sigma_y = (dudyJ + LIFT * (0.5 * u_jump .* nyJ)) ./ J

    sigmaM_n = nx .* (Vf * sigma_x) + ny .* (Vf * sigma_y)
    sigmaP_n = -sigmaM_n[mapP]
    sigmaP_n[md.mapB] .= sigmaM_n[md.mapB]
    sigma_flux = @. 0.5 * (sigmaP_n - sigmaM_n) + u_jump

    d_sigma_x_dx_J = rxJ .* (Dr * sigma_x) + sxJ .* (Ds * sigma_x)
    d_sigma_y_dy_J = ryJ .* (Dr * sigma_y) + syJ .* (Ds * sigma_y)
    div_sigma = d_sigma_x_dx_J + d_sigma_y_dy_J
    du .+= epsilon * (div_sigma + LIFT * (sigma_flux .* Jf)) ./ J

    # add forcing f(x, t) = 1.0
    du .+= 1.0
end

N = 3 # polynomial degree
num_elements = 16

rd = RefElemData(Tri(), N)
md = MeshData(uniform_mesh(Tri(), num_elements), rd)
# md = make_periodic(md)

# impose a discontinuous initial condition
(; x, y) = md
# u = u0.(x, y)

u = zeros(rd.Np, md.num_elements)

epsilon = 0.01
map_inflow = findall(@. abs(md.xf + 1) < 1e-12 || abs(md.yf + 1) < 1e-12)

params = (; rd, md, epsilon, map_inflow)
tspan = (0.0, .50)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 25), 
            callback=AliveCallback(alive_interval=10))

scatter(vec(rd.Vp * x), vec(rd.Vp * y), zcolor=vec(rd.Vp * sol.u[end]), 
        markersize=2, markerstrokewidth=0, legend=false, colorbar=true)

# xp, yp = vec(rd.Vp * x), vec(rd.Vp * y)
# @gif for i in eachindex(sol.u)
#     global xp, yp
#     t = sol.t[i]
#     scatter(xp, yp, zcolor=vec(rd.Vp * sol.u[i]), 
#         markersize=2, markerstrokewidth=0, legend=false)
# end