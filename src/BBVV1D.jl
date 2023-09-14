module BBVV1D

using Printf, WriteVTK

export PointCloud, BondBasedMaterial, VelocityBC, simulation

struct PointCloud
    n_points::Int
    position::Vector{Float64}
    volume::Vector{Float64}
end

struct BondBasedMaterial
    δ::Float64
    bc::Float64
    rho::Float64
    εc::Float64
end

struct VelocityBC
    fun::Function
    point_id_set::Vector{Int}
end

struct SimulationParameters
    mat::BondBasedMaterial
    pc::PointCloud
    cells::Vector{MeshCell{VTKCellType, Tuple{Int64}}}
    n_bonds::Int
    bonds::Vector{Int}
    init_dists::Vector{Float64}
    n_family_members::Vector{Int}
    hood_range::Vector{UnitRange{Int}}
    bcs::Vector{VelocityBC}
    n_timesteps::Int
    export_freq::Int
    export_path::String
end

struct GlobalStorage
    position::Vector{Float64}
    displacement::Vector{Float64}
    velocity::Vector{Float64}
    velocity_half::Vector{Float64}
    acceleration::Vector{Float64}
    b_int::Vector{Float64}
end


"""
MAIN FUNCTION
"""
function simulation(pc::PointCloud, mat::BondBasedMaterial, bcs::Vector{VelocityBC};
                    n_timesteps::Int=1000, export_freq::Int=10,
                    export_path::String="results")
    walltime = @elapsed begin
        sp, gs = init_simulation(pc, mat, bcs, n_timesteps, export_freq, export_path)
        time_loop!(gs, sp)
    end
    open(joinpath(export_path, "logfile.log"), "w+") do io
        write(io, "simulation completed after $walltime seconds (wall time)")
    end
    return sp, gs
end


function init_simulation(pc::PointCloud, mat::BondBasedMaterial, bcs::Vector{VelocityBC},
                         n_timesteps::Int, export_freq::Int, export_path::String)

    print("initialization...")

    # simulation parameters
    bonds, n_bonds, n_family_members, init_dists, hood_range = find_bonds(pc, mat.δ)
    cells = get_cells(pc.n_points)
    sp = SimulationParameters(mat, pc, cells, n_bonds, bonds, init_dists, n_family_members,
                              hood_range, bcs, n_timesteps, export_freq, export_path)

    # global storage of the changing variables
    position = copy(pc.position)
    displacement = zeros(pc.n_points)
    velocity = zeros(pc.n_points)
    velocity_half = zeros(pc.n_points)
    acceleration = zeros(pc.n_points)
    b_int = zeros(pc.n_points)
    gs = GlobalStorage(position, displacement, velocity, velocity_half, acceleration, b_int)

    println("\r✔ initialization   ")

    return sp, gs
end

function time_loop!(gs::GlobalStorage, sp::SimulationParameters)
    Δt = calc_stable_timestep(sp)
    @show Δt
    Δt½ = 0.5Δt
    export_vtk(sp, gs, 0, 0.0)

    print("time loop...")

    for timestep in 1:sp.n_timesteps
        time = timestep * Δt
        update_velocity_half!(gs, sp, Δt½)
        apply_bcs!(gs, sp, time)
        update_disp_and_position!(gs, sp, Δt)
        compute_forcedensity!(gs, sp)
        compute_eq_of_motion!(gs, sp, Δt½)
        if mod(timestep, sp.export_freq) == 0
            export_vtk(sp, gs, timestep, time)
        end
    end

    println("\r✔ time loop   ")

    return nothing
end

function PointCloud(lx::Real, Δx::Real)
    gridx = range(; start = Δx / 2, stop = lx - Δx / 2, step = Δx)
    n_points = length(gridx)
    volume = fill(Δx^3, n_points)
    return PointCloud(n_points, gridx, volume)
end

function find_bonds(pc::PointCloud, δ::Float64)
    bonds = Vector{Int}()
    init_dists = Vector{Float64}()
    n_family_members = zeros(Int, pc.n_points)
    hood_range = fill(0:0, pc.n_points)
    n_bonds = 0
    for i in 1:pc.n_points
        n_neighbors_of_point_i = 0
        hood_range_start = n_bonds + 1
        for j in 1:pc.n_points
            if i !== j
                L = abs(pc.position[j] - pc.position[i])
                if L <= δ
                    n_bonds += 1
                    n_neighbors_of_point_i += 1
                    push!(bonds, j)
                    push!(init_dists, L)
                end
            end
        end
        n_family_members[i] = n_neighbors_of_point_i
        hood_range[i] = hood_range_start:n_bonds
    end
    @assert length(bonds) == n_bonds
    return bonds, n_bonds, n_family_members, init_dists, hood_range
end

function calc_stable_timestep(sp::SimulationParameters)
    timesteps = fill(typemax(Float64), sp.pc.n_points)
    for i in 1:sp.pc.n_points
        dtsum = 0.0
        for cbond in sp.hood_range[i]
            j = sp.bonds[cbond]
            L = sp.init_dists[cbond]
            dtsum += sp.pc.volume[j] * sp.mat.bc / L
        end
        timesteps[i] = sqrt(2 * sp.mat.rho / dtsum)
    end
    Δt = 0.7 * minimum(timesteps)
    return Δt
end

get_cells(n::Int) = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:n]

function export_vtk(sp::SimulationParameters, gs::GlobalStorage, timestep::Int, time::Float64)
    filename = joinpath(sp.export_path, @sprintf("timestep_%04d", timestep))
    vtk_grid(filename, sp.pc.position, sp.cells) do vtk
        vtk["Displacement", VTKPointData()] = gs.displacement
        vtk["Time", VTKFieldData()] = time
    end
    return nothing
end

function compute_forcedensity!(gs::GlobalStorage, sp::SimulationParameters)
    gs.b_int .= 0
    for i in 1:sp.pc.n_points
        for cbond in sp.hood_range[i]
            j = sp.bonds[cbond]
            L = sp.init_dists[cbond]
            Δxij = gs.position[j] - gs.position[i]
            l = abs(Δxij)
            ε = (l - L) / L
            temp = sp.mat.bc * ε / l * sp.pc.volume[j]
            gs.b_int[i] += temp * Δxij
        end
    end
    return nothing
end

function apply_bcs!(gs::GlobalStorage, sp::SimulationParameters, time::Float64)
    for bc in sp.bcs
        value = bc.fun(time)
        @inbounds @simd for i in bc.point_id_set
            gs.velocity_half[i] = value
        end
    end
    return nothing
end

function update_disp_and_position!(gs::GlobalStorage, sp::SimulationParameters, Δt::Float64)
    for i in 1:sp.pc.n_points
        gs.displacement[i] += gs.velocity_half[i] * Δt
        gs.position[i] += gs.velocity_half[i] * Δt
    end
    return nothing
end

function update_velocity_half!(gs::GlobalStorage, sp::SimulationParameters, Δt½::Float64)
    for i in 1:sp.pc.n_points
        gs.velocity_half[i] = gs.velocity[i] + gs.acceleration[i] * Δt½
    end
    return nothing
end

function compute_eq_of_motion!(gs::GlobalStorage, sp::SimulationParameters, Δt½::Float64)
    for i in 1:sp.pc.n_points
        gs.acceleration[i] = gs.b_int[i] / sp.mat.rho
        gs.velocity[i] = gs.velocity_half[i] + gs.acceleration[i] * Δt½
    end
    return nothing
end

end
