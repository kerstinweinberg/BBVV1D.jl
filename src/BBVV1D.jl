module BBVV1D

    include("BBVV_CC.jl")

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

end
