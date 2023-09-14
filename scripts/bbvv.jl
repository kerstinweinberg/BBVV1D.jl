using BBVV1D

RESPATH::String = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results")

function bbvv(simname, l, Δx, vmax, T, nt)
    pc = PointCloud(l, Δx)
    δ = 3.015Δx
    E = 2.1e5
    nu = 0.25
    bc = 18 * E / (3 * (1 - 2 * nu)) / (π * δ^2)
    rho = 8e-6
    εc = 0.01
    mat = BondBasedMaterial(δ, bc, rho, εc)
    set_left = findall(p -> p ≤ Δx, pc.position)
    vwave(t) = t < T ? vmax * sin(2π/T * t) : 0
    bc_left = VelocityBC(vwave, set_left)
    bcs = [bc_left]
    path = joinpath(RESPATH, simname)
    ispath(path) && rm(path; recursive=true, force=true)
    !ispath(path) && mkpath(path) # create the path if it does not exist
    simulation(pc, mat, bcs; n_timesteps=nt, export_freq=10, export_path=path)
    return nothing
end

##--
simname = "bbvv1D"
l = 1.0
Δx = l/1000
v0 = 10
T = 100 * 4.922483834177524e-6
nt = 1000
@time bbvv(simname, l, Δx, v0, T, nt)
