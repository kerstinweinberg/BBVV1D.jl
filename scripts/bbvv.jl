using BBVV1D

RESPATH::String = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results")

# PMMA, SI-Einheiten
const E = 3.2e9       # Pa
const RHO = 1200.0    # kg/m³

function bbvv(simname, l, Δx, vmax, T, totaltime; mat=BBMaterial{EnergySurfaceCorrection}())
    pc = PointCloud(l, Δx)
    δ = 3.015Δx
    params = MaterialParameters(; horizon=δ, rho=RHO, E, alpha=E, epsilon_c=0.01)
    set_left = findall(p -> p ≤ δ, pc.position)   # Last über eine Randschicht der Dicke δ
    vwave(t) = t < T ? vmax * sin(2π/T * t) : 0
    bc_left = VelocityBC(vwave, set_left)
    bcs = [bc_left]
    path = joinpath(RESPATH, simname)
    ispath(path) && rm(path; recursive=true, force=true)
    !ispath(path) && mkpath(path) # create the path if it does not exist
    simulation(pc, params, mat, bcs; totaltime, export_freq=10, export_path=path)
    return nothing
end

##--
simname = "bbvv1D"
l = 1.0                      # m
Δx = l/1000
v0 = 10                      # m/s
c = sqrt(E / RHO)            # Wellengeschwindigkeit √(E/ρ) ≈ 1633 m/s
T = 0.1 * l / c              # Pulsdauer, Pulslänge c·T = l/10
totaltime = 2 * l / c        # ca. 2 Wellendurchläufe (hin und zurück)
@time bbvv(simname, l, Δx, v0, T, totaltime)
# with @btime: 238.600 ms (750902 allocations: 39.90 MiB)
