using OrdinaryDiffEq
using NodesAndModes
using Plots

# polynomial degree
N = 10

# Chebyshev nodes: "good" points for interpolation
x = [-cos(k * pi / N) for k in 0:N] 

# create the nodal differentiation matrix
VDM, dVdx = basis(Line(), N, x) # VDM = V in the notes
D = dVdx / VDM # note that A / B = A * inv(B)

xq, wq = gauss_quad(0, 0, N)
Vq, _ = basis(Line(), N, xq) 
Vq = Vq / VDM # Vq = Vinterp

M = Vq' * diagm(wq) * Vq
Q = M * D

Vf = zeros(2, N+1)
Vf[1, 1] = 1
Vf[2, end] = 1
LIFT = M \ Vf'

function rhs_weak_form!(du, u, parameters, t)
    (; M, Q) = parameters

    # M * du/dt - Q'*u + [-{u}; 0, ..., 0; {u}] = 0
    # du/dt = -inv(M) * (- Q'*u + [-{u}; 0, ..., 0; {u}])
    du .= -Q' * u

    # enforce periodic BCs using central fluxes
    u_left = -0.5 * (u[1] + u[end])
    u_right = 0.5 * (u[1] + u[end])
    du[1] += u_left
    du[end] += u_right

    du .= -(M \ du)
end

# strong form
function rhs_strong_form!(du, u, parameters, t)
    (; D, LIFT) = parameters

    # enforce periodic BCs using central fluxes
    u_flux = [-0.5 * (u[end] - u[1]); 
               0.5 * (u[1] - u[end])]
    du .= -(D * u + LIFT * u_flux)
end

# u0(x) = sin(pi * x)
u0(x) = exp(-25 * x^2)
u0(x) = abs(x) < 0.5

u = u0.(x)
params = (; M, Q, D, LIFT)
tspan = (0.0, 20.0)
ode_weak = ODEProblem(rhs_weak_form!, u, tspan, params)
sol_weak = solve(ode_weak, RK4(), saveat=LinRange(tspan[1], tspan[2], 50))

ode_strong = ODEProblem(rhs_strong_form!, u, tspan, params)
sol_strong = solve(ode_strong, RK4(), saveat=LinRange(tspan[1], tspan[2], 50))

@show maximum(abs.(sol_weak.u[end] - sol_strong.u[end]))