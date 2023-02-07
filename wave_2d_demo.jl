using OrdinaryDiffEq
using StartUpDG
using Plots

N = 3 # polynomial degree
num_elements = 16

rd = RefElemData(Tri(), N)
md = MeshData(uniform_mesh(Tri(), num_elements), rd)
md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters 
    (; Vf, Dr, Ds, LIFT) = rd
    (; rxJ, sxJ, ryJ, syJ, nx, ny, J, Jf, mapP) = md

    p, u, v = view(u, :, :, 1), view(u, :, :, 2), view(u, :, :, 3)

    pM, uM, vM = Vf * p, Vf * u, Vf * v
    pP, uP, vP = view(pM, mapP), view(uM, mapP), view(vM, mapP)

    p_flux = @. 0.5 * ((uP - uM) * nx + (vP - vM) * ny) - 0.5 * (pP - pM)
    u_flux = @. 0.5 * (pP - pM) * nx - 0.5 * (uP - uM)
    v_flux = @. 0.5 * (pP - pM) * ny - 0.5 * (vP - vM)

    dpdr, dpds = Dr * p, Ds * p
    dpdxJ = rxJ .* dpdr + sxJ .* dpds
    dpdyJ = ryJ .* dpdr + syJ .* dpds
    dudxJ = rxJ .* (Dr * u) + sxJ .* (Ds * u)
    dvdyJ = ryJ .* (Dr * v) + syJ .* (Ds * v)
    du[:, :, 1] .= -(dudxJ + dvdyJ + LIFT * (p_flux .* Jf)) ./ J
    du[:, :, 2] .= -(dpdxJ         + LIFT * (u_flux .* Jf)) ./ J
    du[:, :, 3] .= -(dpdyJ         + LIFT * (v_flux .* Jf)) ./ J
end

p0(x, y) = exp(-100 * ((x - 0.25)^2 + y^2)) # assume u0, v0 = 0

(; x, y) = md
u = zeros(rd.Np, md.num_elements, 3)
u[:, :, 1] .= p0.(x, y)

params = (; rd, md)
tspan = (0.0, 2)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines AliveCallback for monitoring progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50), 
            callback=AliveCallback(alive_interval=10))

# scatter(vec(rd.Vp * x), vec(rd.Vp * y), zcolor=vec(rd.Vp * sol.u[end]), 
#         markersize=2, markerstrokewidth=0, legend=false)

xp, yp = vec(rd.Vp * x), vec(rd.Vp * y)        
@gif for i in eachindex(sol.u)    
    scatter(xp, yp, zcolor=vec(rd.Vp * sol.u[i][:,:,1]), 
            markersize=2, markerstrokewidth=0, legend=false, clims=(-.25, .25))
end