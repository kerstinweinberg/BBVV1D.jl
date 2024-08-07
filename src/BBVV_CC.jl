# reference to original file

using Printf, WriteVTK

export PointCloud, BondBasedMaterial, VelocityBC, simulation, read_vtk

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

struct BondBasedMaterial
    δ::Float64
    bc::Float64
    bbconst::Float64
    ccconst::Float64
    E::Float64
    rho::Float64
    εc::Float64
end

struct VelocityBC
    fun::Function
    point_id_set::Vector{Int}
end

"""
MAIN FUNCTION
"""
function simulation(pc::PointCloud, mat::BondBasedMaterial, bcs::Vector{VelocityBC};
                    n_timesteps::Int=0, totaltime=-1e0, export_freq::Int=10,
                    export_path::String="results")
    walltime = @elapsed begin
        print("initialization...")

        neighbor, initial_distance, bond_ids_of_point = find_bonds(pc, mat.δ)
        cells = get_cells(pc.n_points)

        position = copy(pc.position)
        displacement = zeros(pc.n_points)
        velocity = zeros(pc.n_points)
        velocity_half = zeros(pc.n_points)
        acceleration = zeros(pc.n_points)
        barepsilon = zeros(pc.n_points)
        barsigma = zeros(pc.n_points)
        b_int = zeros(pc.n_points)
        println("\r✔ initialization   ")
 
        if totaltime > 0 && n_timesteps > 0
            msg = "Specify either time or number of time steps, not both!"
            throw(ArgumentError(msg))
        end                                                                                # Zeitschritt (CFL)
        Δt = calc_stable_timestep(pc, mat, neighbor, bond_ids_of_point, initial_distance)
        n_timesteps=ceil(Int,totaltime/Δt)
        printstyled(@sprintf("  %i Zeitschritte mit Δt = %6.2e sec\n", n_timesteps, Δt);color=:blue, bold=true)

        print("time loop...")
 
        export_vtk(position, displacement, cells, export_path, 0, 0)

        for timestep in 1:n_timesteps
            time = timestep * Δt

            # update of velocity_half
            for i in 1:pc.n_points
                velocity_half[i] = velocity[i] + acceleration[i] * 0.5Δt
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

            # compute the averaged strain barepsilon at i
            barepsilon .= 0
            for i in 1:pc.n_points
                for current_bond in bond_ids_of_point[i]
                    j = neighbor[current_bond]
                    ΔXij = initial_distance[current_bond]
                    Δuij = displacement[j] - displacement[i]
                    barepsilon[i] += mat.ccconst * Δuij * ΔXij * pc.volume[j] 
                end
            end 

            # Spannung
            barsigma .= mat.E*barepsilon 

             # compute the internal force density b_int
            b_int .= 0
            for i in 1:pc.n_points
                for current_bond in bond_ids_of_point[i]
                    j = neighbor[current_bond]
                    ΔXij = initial_distance[current_bond]
                    epsij = barepsilon[j] + barepsilon[i]
                    b_int[i] += mat.E * mat.ccconst * epsij * ΔXij * pc.volume[j] 
                end
            end

            # back to BB-PD

            # compute the internal force density b_int nach BB
            b_int .= 0
            for i in 1:pc.n_points
                for current_bond in bond_ids_of_point[i]
                    j = neighbor[current_bond]
                    ΔXij = initial_distance[current_bond]
                    Δuij = displacement[j] - displacement[i]
                   b_int[i] += mat.E * mat.bbconst * Δuij / ΔXij * pc.volume[j]  
                end
            end

            # solve the equation of motion
            for i in 1:pc.n_points
                acceleration[i] = b_int[i] / mat.rho
                velocity[i] = velocity_half[i] + acceleration[i] * 0.5Δt
            end

            if mod(timestep, export_freq) == 0
                export_vtk(position, displacement, cells, export_path, timestep, time)
            end
        end

        println("\r✔ time loop   ")
    end

    open(joinpath(export_path, "logfile.log"), "w+") do io
        write(io, "simulation completed after $walltime seconds (wall time)\n\n")
        write(io, "number of points = $(pc.n_points)\n")
        write(io, "number of bonds = $(length(neighbor))\n")
        write(io, "Δt = $Δt\n")
    end

    return nothing
end

function find_bonds(pc::PointCloud, δ::Float64)
    neighbor = Vector{Int}()
    initial_distance = Vector{Float64}()
    bond_ids_of_point = fill(0:0, pc.n_points)
    bond_counter = 0
    for i in 1:pc.n_points
        bond_ids_of_point_start = bond_counter + 1
        for j in 1:pc.n_points
            if i !== j
                L = abs(pc.position[j] - pc.position[i])
                if L <= δ
                    bond_counter += 1
                    push!(neighbor, j)
                    push!(initial_distance, L)
                end
            end
        end
        bond_ids_of_point[i] = bond_ids_of_point_start:bond_counter
    end
    @assert length(neighbor) == bond_counter
    return neighbor, initial_distance, bond_ids_of_point
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

include("VtkReader.jl")
using .VtkReader


