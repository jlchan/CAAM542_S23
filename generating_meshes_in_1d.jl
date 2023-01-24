using OrdinaryDiffEq
using NodesAndModes
using Plots

# polynomial degree
N = 5

# Chebyshev nodes: "good" points for interpolation
r = [-cos(k * pi / N) for k in 0:N] 

# construct the mesh 
num_elements = 4
VX = LinRange(-1, 1, num_elements + 1)
x = zeros(N+1, num_elements)
for e in 1:num_elements
    h = VX[e+1] - VX[e]
    @. x[:,e] = VX[e] + h * 0.5 * (1 + r)
end

scatter(x, sin.(pi * x))
plot!(legend=false)