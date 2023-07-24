using OrdinaryDiffEq
using StartUpDG
using Plots

N = 1 # polynomial degree
num_elements = 32
rd = RefElemData(Tri(), N)
md = MeshData(uniform_mesh(Tri(), num_elements), rd)
md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters 
    (; Vf, Dr, Ds, LIFT) = rd
    (; rxJ, sxJ, ryJ, syJ, J, nx, ny, Jf, mapP) = md

    p, u, v = view(u, :, :, 1), view(u, :, :, 2), view(u, :, :, 3)

    pM, uM, vM = Vf * p, Vf * u, Vf * v
    pP, uP, vP = pM[mapP], uM[mapP], vM[mapP]

    normal_velocity_jump = @. (uP - uM) * nx + (vP - vM) * ny
    p_flux = @. 0.5 * normal_velocity_jump - 0.5 * (pP - pM)
    u_flux = @. 0.5 * (pP - pM) * nx - 0.5 * normal_velocity_jump * nx
    v_flux = @. 0.5 * (pP - pM) * ny - 0.5 * normal_velocity_jump * ny

    dpdr, dpds = Dr * p, Ds * p
    dpdxJ = rxJ .* dpdr + sxJ .* dpds
    dpdyJ = ryJ .* dpdr + syJ .* dpds

    dudxJ = rxJ .* (Dr * u) + sxJ .* (Ds * u)
    dvdyJ = ryJ .* (Dr * v) + syJ .* (Ds * v)

    du[:, :, 1] .= -(dudxJ + dvdyJ + LIFT * (p_flux .* Jf)) ./ J
    du[:, :, 2] .= -(dpdxJ         + LIFT * (u_flux .* Jf)) ./ J
    du[:, :, 3] .= -(dpdyJ         + LIFT * (v_flux .* Jf)) ./ J
end

p_exact(x, y, t) = sin(pi * x) * sin(pi * y) * cos(sqrt(2) * pi * t)
u_exact(x, y, t) = -sqrt(2) / 2 * cos(pi * x) * sin(pi * y) * sin(sqrt(2) * pi * t)
v_exact(x, y, t) = -sqrt(2) / 2 * sin(pi * x) * cos(pi * y) * sin(sqrt(2) * pi * t)

p0(x, y) = p_exact(x, y, 0)

(; x, y) = md
u = zeros(rd.Np, md.num_elements, 3)
u[:, :, 1] .= p0.(x, y)

params = (; rd, md)
tspan = (0.0, .7)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50), 
            callback=AliveCallback(alive_interval=10))

p_error = maximum(abs.(p_exact.(rd.Vp * x, rd.Vp * y, sol.t[end]) .- rd.Vp * sol.u[end][:, :, 1]))
u_error = maximum(abs.(u_exact.(rd.Vp * x, rd.Vp * y, sol.t[end]) .- rd.Vp * sol.u[end][:, :, 2]))
v_error = maximum(abs.(v_exact.(rd.Vp * x, rd.Vp * y, sol.t[end]) .- rd.Vp * sol.u[end][:, :, 3]))

@show p_error, u_error, v_error

# scatter(vec(rd.Vp * x), vec(rd.Vp * y), zcolor=vec(p_exact.(rd.Vp * x, rd.Vp*y, sol.t[end]) .- rd.Vp * sol.u[end][:,:,1]), 
#         markersize=2, markerstrokewidth=0, legend=false, ratio=1)

# xp, yp = vec(rd.Vp * x), vec(rd.Vp * y)        
# @gif for i in eachindex(sol.u)    
#     scatter(xp, yp, zcolor=vec(rd.Vp * sol.u[i][:,:,1]), 
#             markersize=2, markerstrokewidth=0, legend=false, clims=(-.25, .25))
# end