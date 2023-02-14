using OrdinaryDiffEq
using StartUpDG
using Plots

N = 4 # polynomial degree
num_elements = 16

rd = RefElemData(Tri(), N)
md = MeshData(uniform_mesh(Tri(), num_elements), rd)
md = make_periodic(md)

# strong form
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters # rd = parameters.rd, md = parameters.md
    (; Vf, Dr, Ds, LIFT) = rd
    (; rxJ, sxJ, nx, J, Jf, mapP) = md

    uM = Vf * u
    uP = uM[mapP]
    u_flux = @. 0.5 * (uP - uM) * nx - 0.5 * (uP - uM) * abs(nx)

    dudxJ = rxJ .* (Dr * u) + sxJ .* (Ds * u)
    du .= -(dudxJ + LIFT * (u_flux .* Jf)) ./ J
end

u0(x, y) = sin(pi * x) * sin(pi * y)
# u0(x) = exp(-25 * x^2)
# u0(x, y) = sin(pi * x) 

# impose a discontinuous initial condition
(; x, y) = md
# u = u0.(x, y)

mean(x) = sum(x) / length(x)
u = zeros(rd.Np, md.num_elements)
for e in 1:md.num_elements
    yc = mean(y[:,e])
    if abs(yc) < 0.5
        u[:,e] .= u0.(x[:, e], y[:, e])
    end
end


params = (; rd, md)
tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u, tspan, params)

include("alive.jl") # defines an AliveCallback to monitor solve progress
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 25), 
            callback=AliveCallback(alive_interval=10))

# scatter(vec(rd.Vp * x), vec(rd.Vp * y), zcolor=vec(rd.Vp * sol.u[end]), 
#         markersize=2, markerstrokewidth=0, legend=false)

xp, yp = vec(rd.Vp * x), vec(rd.Vp * y)
@gif for i in eachindex(sol.u)
    global xp, yp
    t = sol.t[i]
    scatter(xp, yp, zcolor=vec(rd.Vp * sol.u[i]), 
        markersize=2, markerstrokewidth=0, legend=false)
end