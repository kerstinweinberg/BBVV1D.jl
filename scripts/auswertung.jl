
N=1000000

    # setup
    root = joinpath(@__DIR__, "..", "results", "xwave_1D_v1")
    path = joinpath(root, "xwave_n-$(N)")
    vtk_path = joinpath(path, "vtk")
    post_path = joinpath(path, "post")
    wave_position_data_file = joinpath(post_path, "wavepos.csv")
    wave_speed_summary_file = joinpath(post_path, "wavespeed_summary.txt")
    all_wave_speed_summary_file = joinpath(root, "wavespeed_summary.txt")
    
    # Aufbau Modell und Welle
    printstyled("\n--- PREPROCESSING ---\n", color=:blue, bold=true)
    laenge = 2.0          # Stablaenge [m]
    ΔX = laenge / N
    T, vmax = 5.0e-5, 2.0 # Wellenpuls
    printstyled(@sprintf("  Wellenpuls in %6.2f m langem Stab hat die Amplitude %6.2f m/s fuer T = %6.2f musec (5 cm)\n", laenge, vmax,T*1e+6);color=:blue, bold=true)

    δ = 3.015ΔX
    E = 1000e6     # Polyamid, PE, PVC etc
    rho = 1000.0   # ergibt cL=1000 m/sec
    #
    bbconst = 2/(δ^2)
    bc = E*bbconst
    ccconst = 3/(2*δ^3)
    εc = 0.01
    mat = BondBasedMaterial(δ, bc, bbconst, ccconst, E, rho, εc)
    #    @show E,bbconst, ccconst
    printstyled(@sprintf("  Diskretisierung aus %i Punkten, 1D, der Horizont ist %6.2f mm\n", N,(δ*1e+3));color=:blue, bold=true)
    printstyled(@sprintf("  E = %8.2f MPa\n", (E*1e-6));color=:blue, bold=true)
    printstyled(@sprintf("  ρ = %8.2f g/cm^3\n\n", rho);color=:blue, bold=true)


# Auswertung Geschwindigkeit
printstyled("\n--- POSTPROCESSING ---\n", color=:blue, bold=true)
c_L = sqrt(E/rho)                            #  M=E*(1-nu)/((1+nu)*(1-2*nu))

# finde in jedem File die maximale Verschiebung -  umax, tmax, xmax 
function find_wave_position(r0, r, id)       #  -> wave_position_data_file  
    tmax = first(r[:time])
    tmax < 1.5T && return nothing
    umax, pids = @views findmax(r[:displacement])
    xmax = r[:position][1, pids]
    open(wave_position_data_file, "a+") do io
        @printf(io, "%.12f,%.12f,%.12f\n", tmax, xmax, umax)
    end
    return nothing
end

@time begin
    process_each_export(find_wave_position, vtk_path)
    results = readdlm(wave_position_data_file, ',', Float64)  # Zeilen mit umax
    t, x̂, û = results[:,1], results[:,2], results[:,3]        # Vektoren
    c_w = estimate_velocity(t, x̂, û)                              # aus x-t-Fit
    Δc = c_w - c_L
    Δcp = 100 * Δc / c_L
    printstyled(@sprintf("c_L = %8.2f m/s\n", c_L); color=:green, bold=true)
    printstyled(@sprintf("c_w = %8.2f m/s\n", c_w); color=:blue, bold=true)
    printstyled(@sprintf("Δc  = %8.2f m/s (%.3f %%)\n", Δc, Δcp); color=:red,
                bold=true)
    open(wave_speed_summary_file, "w+") do io
        write(io, @sprintf("""--- WAVE SPEED SUMMARY ---
                            Theoretical wave speed c_L = √(E/ρ):
                                c_L = %13.7f m/s
                            Calculated wave speed in simulation:
                                c_w = %13.7f m/s
                                Δc  = %13.7f m/s (%.3f %%)
                            """, c_L, c_w, Δc, Δcp))
    end
    open(all_wave_speed_summary_file, "a+") do io
        write(io, @sprintf("%s, %13.7f\n", path, c_w))
    end
end