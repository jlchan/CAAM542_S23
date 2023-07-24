using OrdinaryDiffEq
using StartUpDG
using Plots

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md, Dr_weak, Ds_weak, b_x_fun, b_y_fun) = parameters 
    (; Vf, Vq, Pq, Dr, Ds, LIFT) = rd
    (; rxJ, sxJ, ryJ, syJ, nx, ny, J, Jf, mapP) = md

    (; xq, yq, xf, yf) = md
    b_x = b_x_fun.(xq, yq, t)
    b_y = b_y_fun.(xq, yq, t)
    b_n = @. b_x_fun(xf, yf, t) * nx + b_y_fun(xf, yf, t) * ny

    uM = Vf * u
    uP = uM[mapP]
    u_flux = @. 0.5 * (b_n * uP) - 0.5 * abs(b_n) * (uP - uM)
    
    uq = Vq * u
    bxu = Pq * (b_x .* uq)
    byu = Pq * (b_y .* uq)
    dfxdr = 0.5 * (Dr_weak * (bxu) + Pq * (b_x .* (Vq * Dr * u)))
    dfxds = 0.5 * (Ds_weak * (bxu) + Pq * (b_x .* (Vq * Ds * u)))    
    dfydr = 0.5 * (Dr_weak * (byu) + Pq * (b_y .* (Vq * Dr * u)))
    dfyds = 0.5 * (Ds_weak * (byu) + Pq * (b_y .* (Vq * Ds * u)))
    dfxdxJ = rxJ .* dfxdr + sxJ .* dfxds
    dfydyJ = ryJ .* dfydr + syJ .* dfyds
    du .= -(dfxdxJ + dfydyJ + LIFT * (u_flux .* Jf)) ./ J
end

N = 1 # polynomial degree
num_elements = 29

rd = RefElemData(Tri(), N)
(VX, VY), EToV = uniform_mesh(Tri(), num_elements)
@. VX = 0.5 * (1 + VX)
@. VY = 0.5 * (1 + VY)
md = MeshData((VX, VY), EToV, rd)
md = make_periodic(md)

# impose a discontinuous initial condition
(; x, y) = md

#u = @. exp(x * y)
u = @. exp(-100 * ((x - 0.75)^2 + (y - 0.75)^2))

# rotating flow
b_x_fun(x, y, t) = sin(pi * x) * cos(pi * y)  #* cos(pi / 5 * t)
b_y_fun(x, y, t) = -cos(pi * x) * sin(pi * y) #* cos(pi / 5 * t)

# compute weak diff operators
(; M, Dr, Ds) = rd
Dr_weak = M \ (-Dr' * M)
Ds_weak = M \ (-Ds' * M)

params = (; rd, md, Dr_weak, Ds_weak, b_x_fun, b_y_fun)
tspan = (0.0, 10.0)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50), 
            callback=AliveCallback(alive_interval=25))

# create a subtriangulation for plotting            
using Triangulate
triin = Triangulate.TriangulateIO()
triin.pointlist = hcat(rd.rp, rd.sp)'
triout, _ = triangulate("cQ", triin)
tri = triout.trianglelist

using TriplotRecipes: TriPseudocolor
xp, yp = rd.Vp * x, rd.Vp * y

# up = rd.Vp * sol.u[end-15]
# plist = [TriPseudocolor(xp[:,i], yp[:,i], up[:,i], tri) for i in axes(xp, 2)]
# plot(plist, clims=(-.1, .1))

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    up = rd.Vp * sol.u[i]
    plot([TriPseudocolor(xp[:,i], yp[:,i], up[:,i], tri) for i in axes(xp, 2)], clims=extrema(sol.u[1]))
end fps=10