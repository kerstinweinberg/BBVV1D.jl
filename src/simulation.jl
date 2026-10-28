# 1D-Peridynamik, Aufbau angelehnt an Peridynamics.jl (github.com/kaipartmann/Peridynamics.jl):
# Bond / BondSystem / Storage, Materialtypen mit Korrektur-Typparameter, Kernel-Funktionen,
# force_density_point! je Material und Velocity-Verlet mit Δt aus der Bondkonstante bc.

struct PointCloud
    n_points::Int
    position::Vector{Float64}
    volume::Vector{Float64}
end

function PointCloud(lx::Real, Δx::Real)
    gridx = range(; start = Δx / 2, stop = lx - Δx / 2, step = Δx)
    n_points = length(gridx)
    volume = fill(Δx, n_points)
    return PointCloud(n_points, gridx, volume)
end

#-- Materialparameter ---------------------------------------------------------------------

"""
    MaterialParameters(; horizon, rho, E, alpha=E, epsilon_c=0.01)

Parameter aller Modelle. Die Bondkonstante `bc = 2E/δ²` ist das 1D-Gegenstück zu
`bc = 18K/(πδ⁴)` in Peridynamics.jl (aus ½Eε² = ¼ bc ε² δ²); sie bestimmt auch den
Zeitschritt. `alpha` ist der Deviatormodul des OSB-Modells (in 3D 15G).
"""
struct MaterialParameters
    δ::Float64
    rho::Float64
    E::Float64
    α::Float64
    εc::Float64
    bc::Float64
end

function MaterialParameters(; horizon::Real, rho::Real, E::Real, alpha::Real=E,
                            epsilon_c::Real=0.01)
    return MaterialParameters(horizon, rho, E, alpha, epsilon_c, 2 * E / horizon^2)
end

#-- Kernel (wie Peridynamics.jl) ----------------------------------------------------------

@inline const_one_kernel(δ, L) = 1.0
@inline linear_kernel(δ, L) = δ / L
@inline function cubic_b_spline_kernel(δ, L)
    ξ = L / δ
    if 0 < ξ ≤ 0.5
        return 2/3 - 4 * ξ^2 + 4 * ξ^3
    elseif 0.5 < ξ ≤ 1
        return 4/3 - 4 * ξ + 4 * ξ^2 - 4/3 * ξ^3
    else
        return 0.0
    end
end

#-- Korrekturen ---------------------------------------------------------------------------

abstract type AbstractCorrection end

"Keine Korrektur: Kontinuumskonstanten."
struct NoCorrection <: AbstractCorrection end

"""
Energiebasierte Oberflächenkorrektur nach Le & Bobaru (2018), wie in Peridynamics.jl:
mfactor_i = Ψ_theorie / Ψ_pd bei gleichmäßiger Dehnung, Bondfaktor = (mfactor_i + mfactor_j)/2.
Korrigiert in 1D auch den Diskretisierungsfehler der Kontinuumskonstanten.
"""
struct EnergySurfaceCorrection <: AbstractCorrection end

#-- Materialien ---------------------------------------------------------------------------

abstract type AbstractMaterial{C<:AbstractCorrection} end

"""
    BBMaterial()
    BBMaterial{EnergySurfaceCorrection}()

Bond-based: b_i = Σ β bc ε V_j sign(ξ).
"""
struct BBMaterial{C} <: AbstractMaterial{C} end
BBMaterial() = BBMaterial{NoCorrection}()

"""
    OSBMaterial(; kernel=linear_kernel)
    OSBMaterial{EnergySurfaceCorrection}(; kernel=linear_kernel)

Ordinary state-based (LPS, 1D): θ = (1/m) Σ ω L e V, t = ω(c2 L + c1 e) mit
c1 = α/m, c2 = θ (E/m - c1). Default-Kernel wie Peridynamics.jl: linear δ/L.
"""
struct OSBMaterial{C,K} <: AbstractMaterial{C}
    kernel::K
end
OSBMaterial{C}(; kernel::K=linear_kernel) where {C,K} = OSBMaterial{C,K}(kernel)
OSBMaterial(; kwargs...) = OSBMaterial{NoCorrection}(; kwargs...)

"""
    CCMaterial(; calibration=:continuum)

Energieäquivalenz-Modell (Mittelung über ε²):
barepsilon_i = cc Σ Δu L V_j,  b_i = E cc Σ (barepsilon_i + barepsilon_j) L V_j.
`calibration = :continuum` verwendet cc = 3/(2δ³) (Original), `:discrete` cc = 1/√(q M₃)
aus den diskreten Summen im Inneren. Die Le-Bobaru-Korrektur ist nicht anwendbar, weil die
Energie bei gleichmäßiger Dehnung im Inneren verschwindet.
"""
struct CCMaterial <: AbstractMaterial{NoCorrection}
    calibration::Symbol
    function CCMaterial(; calibration::Symbol=:continuum)
        calibration in (:continuum, :discrete) ||
            throw(ArgumentError("calibration must be :continuum or :discrete"))
        return new(calibration)
    end
end

correction_type(::AbstractMaterial{C}) where {C} = C

"""
    default_material(model::Symbol)

Kurzform für Skripte: `:BB` → BB mit Energiekorrektur, `:CC` → CC diskret kalibriert,
`:OSB` (oder `:SB`) → OSB mit linearem Kernel.
"""
function default_material(model::Symbol)
    model === :BB && return BBMaterial{EnergySurfaceCorrection}()
    model === :CC && return CCMaterial(; calibration=:discrete)
    model in (:OSB, :SB) && return OSBMaterial()
    throw(ArgumentError("unknown model $model, use :BB, :CC or :OSB"))
end

#-- Bonds und System ----------------------------------------------------------------------

struct Bond
    neighbor::Int
    length::Float64
end

"""
    BondSystem(pc, mat, params)

Referenzlage, Volumen, Bonds je Punkt und Korrekturfaktor `scfactor` je Bond
(1 ohne Korrektur).
"""
struct BondSystem
    position::Vector{Float64}
    volume::Vector{Float64}
    bonds::Vector{Bond}
    bond_ids::Vector{UnitRange{Int}}
    scfactor::Vector{Float64}
end

@inline each_bond_idx(system::BondSystem, i::Int) = system.bond_ids[i]
@inline each_point_idx(system::BondSystem) = eachindex(system.position)

function BondSystem(pc::PointCloud, mat::AbstractMaterial, params::MaterialParameters)
    bonds, bond_ids = find_bonds(pc, params.δ)
    system = BondSystem(copy(pc.position), copy(pc.volume), bonds, bond_ids,
                        ones(length(bonds)))
    init_correction!(system, mat, params)
    return system
end

# Punkte sind sortiert und äquidistant -> nur ein Indexfenster um i durchsuchen
function find_bonds(pc::PointCloud, δ::Float64)
    bonds = Vector{Bond}()
    bond_ids = fill(0:0, pc.n_points)
    Δx = minimum(pc.volume)
    window = ceil(Int, δ / Δx) + 1
    for i in 1:pc.n_points
        start = length(bonds) + 1
        for j in max(1, i - window):min(pc.n_points, i + window)
            if i !== j
                L = abs(pc.position[j] - pc.position[i])
                L <= δ && push!(bonds, Bond(j, L))
            end
        end
        bond_ids[i] = start:length(bonds)
    end
    return bonds, bond_ids
end

#-- Storage -------------------------------------------------------------------------------

"""
Zustandsgrößen; `barepsilon` und `ccconst` werden nur vom CC-Modell verwendet.
"""
struct Storage
    position::Vector{Float64}
    displacement::Vector{Float64}
    velocity::Vector{Float64}
    velocity_half::Vector{Float64}
    acceleration::Vector{Float64}
    b_int::Vector{Float64}
    strain_energy_density::Vector{Float64}
    barepsilon::Vector{Float64}
    ccconst::Float64
end

function Storage(system::BondSystem, mat::AbstractMaterial, params::MaterialParameters)
    n = length(system.position)
    cc = mat isa CCMaterial ? calc_ccconst(system, mat, params) : NaN
    return Storage(copy(system.position), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
                   zeros(n), zeros(n), cc)
end

#-- Kraftdichten --------------------------------------------------------------------------

function calc_force_density!(storage::Storage, system::BondSystem, mat::AbstractMaterial,
                             params::MaterialParameters)
    storage.b_int .= 0
    for i in each_point_idx(system)
        force_density_point!(storage, system, mat, params, i)
    end
    return nothing
end

function force_density_point!(storage::Storage, system::BondSystem, ::BBMaterial,
                              params::MaterialParameters, i)
    (; position, b_int) = storage
    (; bonds, volume, scfactor) = system
    for bond_id in each_bond_idx(system, i)
        bond = bonds[bond_id]
        j, L = bond.neighbor, bond.length
        Δxij = position[j] - position[i]
        l = abs(Δxij)
        ε = (l - L) / L
        b_int[i] += scfactor[bond_id] * params.bc * ε * volume[j] * Δxij / l
    end
    return nothing
end

function force_density_point!(storage::Storage, system::BondSystem, mat::OSBMaterial,
                              params::MaterialParameters, i)
    wvol = calc_weighted_volume(system, mat, params, i)
    iszero(wvol) && return nothing
    dil = calc_dilatation(storage, system, mat, params, wvol, i)
    (; position, b_int) = storage
    (; bonds, volume, scfactor) = system
    c1 = params.α / wvol
    c2 = dil * (params.E / wvol - c1)
    for bond_id in each_bond_idx(system, i)
        bond = bonds[bond_id]
        j, L = bond.neighbor, bond.length
        Δxij = position[j] - position[i]
        l = abs(Δxij)
        ω = mat.kernel(params.δ, L) * scfactor[bond_id]
        p = ω * (c2 * L + c1 * (l - L)) / l * Δxij
        b_int[i] += p * volume[j]
        b_int[j] -= p * volume[i]
    end
    return nothing
end

function calc_weighted_volume(system::BondSystem, mat::OSBMaterial,
                              params::MaterialParameters, i)
    wvol = 0.0
    for bond_id in each_bond_idx(system, i)
        bond = system.bonds[bond_id]
        ω = mat.kernel(params.δ, bond.length) * system.scfactor[bond_id]
        wvol += ω * bond.length^2 * system.volume[bond.neighbor]
    end
    return wvol
end

function calc_dilatation(storage::Storage, system::BondSystem, mat::OSBMaterial,
                         params::MaterialParameters, wvol, i)
    dil = 0.0
    for bond_id in each_bond_idx(system, i)
        bond = system.bonds[bond_id]
        j, L = bond.neighbor, bond.length
        l = abs(storage.position[j] - storage.position[i])
        ω = mat.kernel(params.δ, L) * system.scfactor[bond_id]
        dil += ω * L * (l - L) * system.volume[j]
    end
    return dil / wvol
end

# CC braucht barepsilon aller Nachbarn -> zuerst barepsilon für alle Punkte
function calc_force_density!(storage::Storage, system::BondSystem, mat::CCMaterial,
                             params::MaterialParameters)
    (; displacement, barepsilon, ccconst) = storage
    (; bonds, volume) = system
    for i in each_point_idx(system)
        barepsilon[i] = 0.0
        for bond_id in each_bond_idx(system, i)
            bond = bonds[bond_id]
            j = bond.neighbor
            barepsilon[i] += ccconst * (displacement[j] - displacement[i]) * bond.length *
                             volume[j]
        end
    end
    storage.b_int .= 0
    for i in each_point_idx(system)
        force_density_point!(storage, system, mat, params, i)
    end
    return nothing
end

function force_density_point!(storage::Storage, system::BondSystem, ::CCMaterial,
                              params::MaterialParameters, i)
    (; barepsilon, ccconst, b_int) = storage
    (; bonds, volume) = system
    for bond_id in each_bond_idx(system, i)
        bond = bonds[bond_id]
        j = bond.neighbor
        epsij = barepsilon[i] + barepsilon[j]
        b_int[i] += params.E * ccconst * epsij * bond.length * volume[j]
    end
    return nothing
end

# cc = 3/(2δ³) (Original) oder 1/√(q M₃) mit q = Σ L V, M₃ = Σ L³ V im Inneren (Maximum)
function calc_ccconst(system::BondSystem, mat::CCMaterial, params::MaterialParameters)
    mat.calibration === :continuum && return 3 / (2 * params.δ^3)
    q, M3 = 0.0, 0.0
    for i in each_point_idx(system)
        qi, M3i = 0.0, 0.0
        for bond_id in each_bond_idx(system, i)
            bond = system.bonds[bond_id]
            qi += bond.length * system.volume[bond.neighbor]
            M3i += bond.length^3 * system.volume[bond.neighbor]
        end
        q, M3 = max(q, qi), max(M3, M3i)
    end
    return 1 / sqrt(q * M3)
end

#-- Verzerrungsenergiedichte (für die Oberflächenkorrektur) -------------------------------

function strain_energy_density_point!(storage::Storage, system::BondSystem, ::BBMaterial,
                                      params::MaterialParameters, i)
    (; position, strain_energy_density) = storage
    (; bonds, volume, scfactor) = system
    Ψ = 0.0
    for bond_id in each_bond_idx(system, i)
        bond = bonds[bond_id]
        j, L = bond.neighbor, bond.length
        ε = (abs(position[j] - position[i]) - L) / L
        Ψ += 0.25 * scfactor[bond_id] * params.bc * ε * ε * L * volume[j]
    end
    strain_energy_density[i] = Ψ
    return nothing
end

function strain_energy_density_point!(storage::Storage, system::BondSystem,
                                      mat::OSBMaterial, params::MaterialParameters, i)
    (; position, strain_energy_density) = storage
    (; bonds, volume, scfactor) = system
    wvol = calc_weighted_volume(system, mat, params, i)
    iszero(wvol) && return nothing
    dil = calc_dilatation(storage, system, mat, params, wvol, i)
    Ψdev = 0.0
    for bond_id in each_bond_idx(system, i)
        bond = bonds[bond_id]
        j, L = bond.neighbor, bond.length
        edev = (abs(position[j] - position[i]) - L) - dil * L
        ω = mat.kernel(params.δ, L) * scfactor[bond_id]
        Ψdev += params.α / (2 * wvol) * ω * edev * edev * volume[j]
    end
    strain_energy_density[i] = 0.5 * params.E * dil^2 + Ψdev
    return nothing
end

init_correction!(::BondSystem, ::AbstractMaterial{NoCorrection}, ::MaterialParameters) =
    nothing

# Le & Bobaru (2018), in 1D: nur eine Raumrichtung, Bondfaktor = Mittel von i und j
function init_correction!(system::BondSystem, mat::AbstractMaterial{EnergySurfaceCorrection},
                          params::MaterialParameters)
    storage = Storage(system, mat, params)
    λ = 1.001 # deformation stretch factor
    storage.position .= λ .* system.position
    mfactor = zeros(length(system.position))
    Ψ_theory = 0.5 * params.E * (λ - 1)^2
    for i in each_point_idx(system)
        strain_energy_density_point!(storage, system, mat, params, i)
        mfactor[i] = Ψ_theory / storage.strain_energy_density[i]
    end
    for i in each_point_idx(system)
        for bond_id in each_bond_idx(system, i)
            j = system.bonds[bond_id].neighbor
            system.scfactor[bond_id] = (mfactor[i] + mfactor[j]) / 2
        end
    end
    return nothing
end

#-- Zeitintegration -----------------------------------------------------------------------

struct VelocityBC
    fun::Function
    point_id_set::Vector{Int}
end

# wie Peridynamics.jl: Δt = safety_factor · min_i sqrt(2ρ / Σ V_j bc / L) für alle Materialien
function calc_stable_timestep(system::BondSystem, params::MaterialParameters;
                              safety_factor::Real=0.7)
    Δt = typemax(Float64)
    for i in each_point_idx(system)
        dtsum = 0.0
        for bond_id in each_bond_idx(system, i)
            bond = system.bonds[bond_id]
            dtsum += system.volume[bond.neighbor] * params.bc / bond.length
        end
        Δt = min(Δt, sqrt(2 * params.rho / dtsum))
    end
    return safety_factor * Δt
end

"""
MAIN FUNCTION

    simulation(pc, params, mat, bcs; n_timesteps=0, totaltime=-1.0, ...)

`mat` ist `BBMaterial`, `CCMaterial` oder `OSBMaterial` (siehe auch `default_material`).
Entweder `n_timesteps` oder `totaltime` angeben, nicht beides.

- `Δt > 0` gibt den Zeitschritt fest vor (z. B. gleich für alle Modelle), sonst wie in
  Peridynamics.jl aus `bc` mit `safety_factor`
- `vtk = false` schreibt keine VTK-Dateien und kein Logfile
- `history = true` gibt alle `export_freq` Schritte die Felder als NamedTuple zurück:
  `(; time, position, displacement, velocity, b_int)`, Matrizen mit n_points × n_out
"""
function simulation(pc::PointCloud, params::MaterialParameters, mat::AbstractMaterial,
                    bcs::Vector{VelocityBC}; n_timesteps::Int=0, totaltime=-1e0, Δt=-1e0,
                    safety_factor::Real=0.7, export_freq::Int=10,
                    export_path::String="results", vtk::Bool=true, history::Bool=false)
    if totaltime > 0 && n_timesteps > 0
        throw(ArgumentError("Specify either time or number of time steps, not both!"))
    elseif totaltime ≤ 0 && n_timesteps ≤ 0
        throw(ArgumentError("Specify either totaltime or n_timesteps!"))
    end

    walltime = @elapsed begin
        print("initialization...")
        system = BondSystem(pc, mat, params)
        storage = Storage(system, mat, params)
        (; position, displacement, velocity, velocity_half, acceleration, b_int) = storage
        cells = get_cells(pc.n_points)
        println("\r✔ initialization   ")

        Δt_stable = calc_stable_timestep(system, params; safety_factor)
        if Δt > 0
            Δt > Δt_stable &&
                @warn @sprintf("Δt = %6.2e größer als stabiler Schritt %6.2e", Δt, Δt_stable)
        else
            Δt = Δt_stable
        end
        if totaltime > 0
            n_timesteps = ceil(Int, totaltime / Δt)
        end
        printstyled(@sprintf("  %s, %i Zeitschritte mit Δt = %6.2e sec\n", mat,
                             n_timesteps, Δt); color=:blue, bold=true)

        print("time loop...")

        vtk && export_vtk(position, displacement, cells, export_path, 0, 0)
        hist = (; time=Float64[], position=pc.position, displacement=Vector{Float64}[],
                velocity=Vector{Float64}[], b_int=Vector{Float64}[])
        history && record_history!(hist, 0.0, displacement, velocity, b_int)

        Δt½ = 0.5 * Δt
        for timestep in 1:n_timesteps
            time = timestep * Δt

            # update of velocity_half
            for i in 1:pc.n_points
                velocity_half[i] = velocity[i] + acceleration[i] * Δt½
            end

            # apply the boundary conditions
            for bc in bcs
                value = bc.fun(time)
                @inbounds @simd for i in bc.point_id_set
                    velocity_half[i] = value
                end
            end

            # update of displacement and position
            for i in 1:pc.n_points
                displacement[i] += velocity_half[i] * Δt
                position[i] += velocity_half[i] * Δt
            end

            # compute the internal force density b_int
            calc_force_density!(storage, system, mat, params)

            # solve the equation of motion
            for i in 1:pc.n_points
                acceleration[i] = b_int[i] / params.rho
                velocity[i] = velocity_half[i] + acceleration[i] * Δt½
            end

            if mod(timestep, export_freq) == 0
                vtk && export_vtk(position, displacement, cells, export_path, timestep, time)
                history && record_history!(hist, time, displacement, velocity, b_int)
            end
        end

        println("\r✔ time loop   ")
    end

    if vtk
        open(joinpath(export_path, "logfile.log"), "w+") do io
            write(io, "simulation completed after $walltime seconds (wall time)\n\n")
            write(io, "material = $mat\n")
            write(io, "number of points = $(pc.n_points)\n")
            write(io, "number of bonds = $(length(system.bonds))\n")
            write(io, "Δt = $Δt\n")
        end
    end

    history || return nothing
    return (; time=hist.time, position=hist.position,
            displacement=reduce(hcat, hist.displacement),
            velocity=reduce(hcat, hist.velocity), b_int=reduce(hcat, hist.b_int))
end

function record_history!(hist, time, displacement, velocity, b_int)
    push!(hist.time, time)
    push!(hist.displacement, copy(displacement))
    push!(hist.velocity, copy(velocity))
    push!(hist.b_int, copy(b_int))
    return nothing
end

get_cells(n::Int) = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:n]

function export_vtk(position, displacement, cells, export_path, timestep, time)
    filename = joinpath(export_path, @sprintf("timestep_%04d", timestep))
    vtk_grid(filename, position, cells) do vtk
        vtk["displacement", VTKPointData()] = displacement
        vtk["time", VTKFieldData()] = time
    end
    return nothing
end
