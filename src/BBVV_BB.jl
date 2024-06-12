

using Printf, WriteVTK 

export PointCloud, BondBasedMaterial, VelocityBC, simulation

struct PointCloud
    n_points::Int
    position::Vector{Float64}
    volume::Vector{Float64}
end

function PointCloud(lx::Real, Δx::Real)
    gridx = range(; start = Δx / 2, stop = lx - Δx / 2, step = Δx)
    n_points = length(gridx)
    volume = fill(Δx, n_points)  # Laenge * A=1
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
                    n_timesteps::Int=1000, export_freq::Int=10,
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
        b_int = zeros(pc.n_points)

        println("\r✔ initialization   ")
        print("time loop...")

        Δt = calc_stable_timestep(pc, mat, neighbor, bond_ids_of_point, initial_distance)

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

            # compute the internal force density b_int
            b_int .= 0
            for i in 1:pc.n_points
                for current_bond in bond_ids_of_point[i]
                    j = neighbor[current_bond]
                    ΔXij = initial_distance[current_bond]
                    Δuij = displacement[j] - displacement[i]
                    b_int[i] += mat.E * mat.bbc * Δuij / ΔXij * pc.volume[j]  
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

function calc_stable_timestep(pc, mat, neighbor, bond_ids_of_point, initial_distance)
    timesteps = fill(typemax(Float64), pc.n_points)
    for i in 1:pc.n_points
        dtsum = 0.0
        for current_bond in bond_ids_of_point[i]
            j = neighbor[current_bond]
            L = initial_distance[current_bond]
            dtsum += pc.volume[j] * mat.bc / L
        end
        timesteps[i] = sqrt(2 * mat.rho / dtsum)
    end
    Δt = 0.7 * minimum(timesteps)
    return Δt
end

get_cells(n::Int) = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:n]

function export_vtk(position, displacement, cells, export_path, timestep, time)
    filename = joinpath(export_path, @sprintf("timestep_%04d", timestep))
    vtk_grid(filename, position, cells) do vtk
        vtk["Displacement", VTKPointData()] = displacement
        vtk["Time", VTKFieldData()] = time
    end
    return nothing
end


