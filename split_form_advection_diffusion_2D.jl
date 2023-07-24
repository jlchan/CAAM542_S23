using OrdinaryDiffEq
using StartUpDG
using Plots

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md, u_BC, b_x_fun, b_y_fun) = parameters 
    (; Vf, Dr, Ds, LIFT) = rd
    (; x, y, rxJ, sxJ, ryJ, syJ, nxJ, nyJ, nx, ny, J, Jf, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    
    dudr = Dr * u
    duds = Ds * u

    # advection part
    b_x = b_x_fun.(x, y, t)
    b_y = b_y_fun.(x, y, t)
    b_n = (Vf * b_x) .* nx + (Vf * b_y) .* ny
    uP[map_inflow] = 2 * u_BC[map_inflow] - uM[map_inflow]
    u_jump = uP - uM
    u_flux = @. 0.5 * u_jump * b_n - 0.5 * abs(b_n) * u_jump
    dudxJ = rxJ .* dudr + sxJ .* duds
    dudyJ = ryJ .* dudr + syJ .* duds
    du .= -(b_x .* dudxJ + b_y .* dudyJ + LIFT * (u_flux .* Jf)) ./ J

    # diffusion part
    @. uP[md.mapB] = 2 * u_BC[md.mapB] - uM[md.mapB] # enforce {u} = 0 => u⁺ + u⁻ = 0 => u⁺ = -u⁻
    @. u_jump = uP - uM
    sigma_x = (dudxJ + LIFT * (0.5 * u_jump .* nxJ)) ./ J
    sigma_y = (dudyJ + LIFT * (0.5 * u_jump .* nyJ)) ./ J

    sigmaM_n = nx .* (Vf * sigma_x) + ny .* (Vf * sigma_y)
    sigmaP_n = -sigmaM_n[mapP] # flip sign for exterior normal
    sigmaP_n[md.mapB] .= sigmaM_n[md.mapB] # reset exterior value of sigmaP_n to enforce σ⁺ = σ⁻.
    sigma_flux = @. 0.5 * (sigmaP_n - sigmaM_n) + u_jump

    d_sigma_x_dx_J = rxJ .* (Dr * sigma_x) + sxJ .* (Ds * sigma_x)
    d_sigma_y_dy_J = ryJ .* (Dr * sigma_y) + syJ .* (Ds * sigma_y)
    div_sigma = d_sigma_x_dx_J + d_sigma_y_dy_J
    du .+= epsilon * (div_sigma + LIFT * (sigma_flux .* Jf)) ./ J
end

N = 3 # polynomial degree
num_elements = 32

rd = RefElemData(Tri(), N)
md = MeshData(uniform_mesh(Tri(), num_elements), rd)
# md = make_periodic(md)

# impose a discontinuous initial condition
(; x, y) = md

u = zeros(rd.Np, md.num_elements)

epsilon = 0.001

# rotating flow
b_x_fun(x, y, t) = 2 * y * (1 - x^2)
b_y_fun(x, y, t) = -2 * x * (1 - y^2)

# compute inflow = domain boundary where b ⋅ n < 0
(; x, y, nx, ny) = md
b_n = (rd.Vf * b_x_fun.(x, y, 0.0)) .* nx + (rd.Vf * b_y_fun.(x, y, 0.0)) .* ny
map_inflow = md.mapB[findall(b_n[md.mapB] .< eps())]

# boundary data: zero everywhere except at x = 1
u_BC = zeros(size(md.xf))
u_BC[findall(@. abs(md.xf - 1) < 1e-12)] .= 1

params = (; rd, md, epsilon, b_x_fun, b_y_fun, u_BC, map_inflow)
tspan = (0.0, 5.0)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 100), 
            callback=AliveCallback(alive_interval=25))

# create a subtriangulation for plotting            
using Triangulate
triin = Triangulate.TriangulateIO()
triin.pointlist = hcat(rd.rp, rd.sp)'
triout, _ = triangulate("cQ", triin)
tri = triout.trianglelist

using TriplotRecipes: TriPseudocolor
xp, yp = rd.Vp * x, rd.Vp * y

up = rd.Vp * sol.u[end]
plist = [TriPseudocolor(xp[:,i], yp[:,i], up[:,i], tri) for i in axes(xp, 2)]
plot(plist)

# @gif for i in eachindex(sol.u)
#     t = sol.t[i]
#     up = rd.Vp * sol.u[i]
#     plot([TriPseudocolor(xp[:,i], yp[:,i], up[:,i], tri) for i in axes(xp, 2)], clims=extrema(sol.u[end]))
# end fps=10