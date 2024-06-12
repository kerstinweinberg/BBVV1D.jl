using BBVV1D, DelimitedFiles, Printf

function xwave(l, N, T, vmax, path)
    Δx = l / N
    pc = PointCloud(l, Δx)
    δ = 3.015Δx
    E = 1000e6     # Polyamid, PE, PVC etc
    rho = 1000.0   # ergibt cL=1000 m/sec
    #
    bbconst = 2/(δ^2)
    bc = E*bbconst
    ccconst = 3/(2*δ^3)
    εc = 0.01
    mat = BondBasedMaterial(δ, bc, bbconst, ccconst, E, rho, εc)
    #
    set_left = findall(p -> p ≤ Δx, pc.position)
    vwave(t) = t < T ? vmax * sin(2π/T * t) : 0
    bc_left = VelocityBC(vwave, set_left)
    bcs = [bc_left]
    #
    simulation(pc, mat, bcs; n_timesteps=1000, export_freq=10, export_path=path)
    return nothing
end

function process_each_export(f::F, vtk_path::AbstractString) where {F<:Function}
    vtk_files = find_vtk_files(vtk_path)
    process_each_export_serial(f, vtk_files)
    return nothing
end

function find_vtk_files(path::AbstractString)
    isdir(path) || throw(ArgumentError("invalid path $path specified!\n"))
    all_files = readdir(path; join=true)
    pvtu_files = filter(x -> endswith(x, ".vtu"), all_files)
    isempty(pvtu_files) && throw(ArgumentError("no pvtu-files in path: $path\n"))
    return pvtu_files
end

function process_step(f::F, ref_result::Dict{Symbol,T}, file::AbstractString,
                      file_id::Int) where {F<:Function,T}
    result = read_vtk(file)
    try
        f(ref_result, result, file_id)
    catch err
        @error "something wrong while processing file $(basename(file))" error=err
    end
    return nothing
end

function process_each_export_serial(f::F, vtk_files::Vector{String}) where {F<:Function}
    ref_result = read_vtk(first(vtk_files))
    for (file_id, file) in enumerate(vtk_files)
        process_step(f, ref_result, file, file_id)
    end
    return nothing
end

function calc_velocity(t, x_w, u_w)
    u_0 = first(u_w)
    valid_until = findfirst(x -> !isapprox(u_0, x; rtol=0.01), u_w)
    if !isnothing(valid_until)
        x_w = x_w[1:valid_until-1]
        t = t[1:valid_until-1]
    end
    n = length(t)
    @assert n == length(x_w)
    t̄ = sum(t) / n
    x̄ = sum(x_w) / n
    v = sum((t .- t̄) .* (x_w .- x̄)) / sum((t .- t̄) .^ 2)
    return v
end

function main(N::Int=10000)

    # setup
    root = joinpath(@__DIR__, "..", "results", "xwave_1D_v1")
    path = joinpath(root, "xwave_n-$(N)")
    vtk_path = joinpath(path, "vtk")
    post_path = joinpath(path, "post")
    wave_position_data_file = joinpath(post_path, "wavepos.csv")
    wave_speed_summary_file = joinpath(post_path, "wavespeed_summary.txt")
    all_wave_speed_summary_file = joinpath(root, "wavespeed_summary.txt")
    lx = 0.2
    ΔX = lx / N
    T, vmax = 1.0e-5, 2.0

    rm(root; recursive=true, force=true)
    mkpath(vtk_path)
    mkpath(post_path)

    printstyled("--- XWAVE WITH N=$(N) ---\n", bold=true, color=:blue)
    @time xwave(lx, N, T, vmax, vtk_path)

    # postprocessing
    printstyled("--POSTPROCESSING--\n", color=:blue, bold=true)
    function find_wave_position(r0, r, id)
        t = first(r[:time])
        t < 1.5T && return nothing
        û, pids = @views findmax(r[:displacement])
        x̂ = r[:position][1, pids]
        open(wave_position_data_file, "a+") do io
            @printf(io, "%.12f,%.12f,%.12f\n", t, x̂, û)
        end
        return nothing
    end
    @time begin
        process_each_export(find_wave_position, vtk_path)
        results = readdlm(wave_position_data_file, ',', Float64)
        t, x̂, û = results[:,1], results[:,2], results[:,3]
        c_0 = 1000
        c_w = calc_velocity(t, x̂, û)
        Δc = c_w - c_0
        Δcp = 100 * Δc / c_0
        printstyled(@sprintf("c_0 = %8.2f m/s\n", c_0); color=:green, bold=true)
        printstyled(@sprintf("c_w = %8.2f m/s\n", c_w); color=:blue, bold=true)
        printstyled(@sprintf("Δc  = %8.2f m/s (%.3f %%)\n", Δc, Δcp); color=:red,
                    bold=true)
        open(wave_speed_summary_file, "w+") do io
            write(io, @sprintf("""--- WAVE SPEED SUMMARY ---
                                Theoretical wave speed c_0 = √(E/ρ):
                                    c_0 = %13.7f m/s
                                Calculated wave speed in simulation:
                                    c_w = %13.7f m/s
                                    Δc  = %13.7f m/s (%.3f %%)
                                """, c_0, c_w, Δc, Δcp))
        end
        open(all_wave_speed_summary_file, "a+") do io
            write(io, @sprintf("%s, %13.7f\n", path, c_w))
        end
    end

    return nothing
end

main()
