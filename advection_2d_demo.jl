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
    (; rd, md) = parameters # rd = parameters.rd, md = parameters.md
    (; Vf, Dr, Ds, LIFT) = rd
    (; rxJ, sxJ, nx, J, Jf, mapP) = md

    uM = Vf * u
    uP = uM[mapP]

    u_flux = @. 0.5 * (uP - uM) * nx - 0.5 * (uP - uM) * abs(nx)
    dudxJ = rxJ .* (Dr * u) + sxJ .* (Ds * u)
    du .= -(dudxJ + LIFT * (u_flux .* Jf)) ./ J
end

# u0(x, y) = sin(pi * x) * sin(pi * y)
# u0(x) = exp(-25 * x^2)
u0(x, y) = sin(pi * x) * (abs(y) < 0.5)

(; xq, yq) = md
u = rd.Pq * u0.(xq, yq)
params = (; rd, md)
tspan = (0.0, 2)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50))

scatter(vec(rd.Vp * x), vec(rd.Vp * y), zcolor=vec(rd.Vp * sol.u[end]), 
        markersize=2, markerstrokewidth=0, legend=false)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    scatter(vec(rd.Vp * x), vec(rd.Vp * y), zcolor=vec(rd.Vp * sol.u[i]), 
        markersize=2, markerstrokewidth=0, legend=false)
end