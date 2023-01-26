using OrdinaryDiffEq
using NodesAndModes
using Plots

# polynomial degree
N = 5

# Chebyshev nodes: "good" points for interpolation
r = [-cos(k * pi / N) for k in 0:N] 

# create nodal matrices
VDM, dVdx = basis(Line(), N, r) # VDM = V in the notes
D = dVdx / VDM # note that A / B = A * inv(B)
rq, wq = gauss_quad(0, 0, N)
Vq, _ = basis(Line(), N, rq) 
Vq = Vq / VDM # Vq = Vp
M = Vq' * diagm(wq) * Vq
Q = M * D

# face interpolation matrix
Vf = zeros(2, N+1)
Vf[1, 1] = 1
Vf[2, end] = 1
LIFT = M \ Vf'

# construct the mesh 
num_elements = 40
VX = LinRange(-1, 1, num_elements + 1)
h = VX[2] - VX[1] # assume h is the same for all elements
x = zeros(N+1, num_elements)
for e in 1:num_elements    
    @. x[:,e] = VX[e] + h * 0.5 * (1 + r)
end
J = h / 2 * ones(N+1, num_elements)

# outward normals on each element
nx = zeros(2, num_elements)
nx[1, :] .= -1
nx[2, :] .= 1

# create node mappings
mapM = reshape(1:2*num_elements, 2, num_elements)
mapP = copy(mapM)
mapP[2, 1] = mapM[1, 2]
mapP[1, num_elements] = mapM[2, num_elements-1]
for e in 1:num_elements    
    if e > 1 && e < num_elements
        mapP[1, e] = mapM[2, e-1]
        mapP[2, e] = mapM[1, e+1]
    end
end
# periodicity
mapP[1, 1] = mapM[2, num_elements]
mapP[2, num_elements] = mapM[1, 1]

flux(uP, uM, nx) = 0.5 * (uP + uM) * nx - 0.5 * (uP - uM)

# strong form
function rhs!(du, u, parameters, t)
    (; D, LIFT, nx, mapP) = parameters

    uM = Vf * u
    uP = uM[mapP]

    # enforce periodic BCs using central fluxes
    u_flux = flux.(uP, uM, nx) - uM .* nx
    du .= -(D * u + LIFT * u_flux) ./ J
end

# u0(x) = sin(pi * x)
u0(x) = exp(-25 * x^2)
u0(x) = abs(x) < 0.5

u = u0.(x)
params = (; D, LIFT, nx, mapP)
tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 50))

# interpolation to equispaced plotting nodes
x̃ = LinRange(-1, 1, 200)
Ṽ, _ = basis(Line(), N, x̃) 
Vp = Ṽ / VDM

# # @show Linf_error = maximum(abs.(Vp * sol.u[end] - u0.(xp)))
@gif for i in eachindex(sol.u)
    t = sol.t[i]
    plot(Vp * x, Vp * sol.u[i])
    plot!(Vp * x, u0.(Vp * x))
    plot!(ylims = extrema(u0.(r)) .+ (-0.5, 0.5), leg=false, title="Time = $t")
end