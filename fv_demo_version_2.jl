using OrdinaryDiffEq
using Plots

N = 100
vertices = LinRange(-1, 1, N+1)

# grid spacing
h = vertices[2] - vertices[1] 

# cell centers
x = 0.5 * (vertices[1:end-1] + vertices[2:end]) 
x = vec(x)'

# create initial condition
# u0(x) = sin(pi * x)
u0(x) = exp(-25 * sin(x)^2)
u = u0.(x) 

flux(u, uP, nx) = 0.5 * (uP + u) * nx - 0.5 * (uP - u)

# matrix to evaluate solution on interfaces
Vf = [1; 1]

# create interface node mappings
num_interfaces = 2 * length(u)
mapM = reshape(1:num_interfaces, 2, :) # "interior" indices
mapP = copy(mapM) # "exterior" indices
for e in 1:size(mapM, 2)
    if 1 < e < size(mapM, 2)
        mapP[1, e] = mapM[2, e-1]
        mapP[2, e] = mapM[1, e+1]
    end
end
mapP[:, 1] = [mapM[2, end]; mapM[1, 2]]
mapP[:, end] = [mapM[2, end-1]; mapM[1, 1]]

# outward normals
nx = [-ones(size(mapM, 2)) ones(size(mapM, 2))]' 

function rhs!(du, u, parameters, t)
    (; h, nx, Vf, mapP) = parameters
    uf = Vf * u
    uP = uf[mapP]
    f = flux.(uf, uP, nx)
    du .= -sum(f, dims=1) ./ h
end

tspan = (0, 4.0)
params = (; h, nx, mapP, Vf)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan[1], tspan[2], 50))

plot(vec(x), vec(sol.u[end]))
plot!(vec(x), vec(u0.(x)), leg=false)

@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(vec(x), vec(sol.u[i]))
    plot!(vec(x), vec(u0.(x)))
    plot!(ylims = extrema(u0.(x)) .+ (-0.5, 0.5), leg=false, title="Time = $t")
end