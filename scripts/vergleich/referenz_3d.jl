# 3D-Referenzlauf mit Peridynamics.jl (github.com/kaipartmann/Peridynamics.jl):
# dünner Stab nach dem Tutorial `wave_in_bar`, u_x über den Querschnitt gemittelt, damit er
# sich mit denselben Plotfunktionen wie die 1D-Modelle vergleichen lässt.

import Peridynamics

"""
    stab_3d(mat, label; l, N, npyz=4, E, rho, nu=0.25, vmax, T, tend, freq=5, path)

Rechnet (oder liest, falls `path` schon Ergebnisse mit denselben Parametern enthält) einen
Stab der Länge `l` mit `N` Punkten in x-Richtung und `npyz × npyz` im Querschnitt. Rückgabe
wie ein 1D-run: `(; label, hist, volume, rho)` mit x ∈ (0, l).
"""
function stab_3d(mat, label; l, N, npyz=4, E, rho, nu=0.25, vmax, T, tend, freq=5, path)
    Δx = l / N
    vtk_path = joinpath(path, "vtk")
    kennung = string((; mat=typeof(mat), l, N, npyz, E, rho, nu, vmax, T, tend, freq))
    kennung_datei = joinpath(path, "parameter.txt")
    gleich = isfile(kennung_datei) && read(kennung_datei, String) == kennung
    if !gleich || !isdir(vtk_path) || isempty(readdir(vtk_path))
        rm(path; recursive=true, force=true)
        pos, vol = Peridynamics.uniform_box(l, npyz * Δx, npyz * Δx, Δx)
        body = Peridynamics.Body(mat, pos, vol)
        Peridynamics.material!(body; horizon=3.015Δx, rho, E, nu)
        # Last über eine Randschicht der Dicke δ (wie in 1D)
        Peridynamics.point_set!(p -> p[1] + l / 2 < 3.015Δx, body, :left)
        Peridynamics.velocity_bc!(t -> 0 ≤ t < T ? vmax * sin(2π / T * t) : 0.0, body,
                                  :left, :x)
        vv = Peridynamics.VelocityVerlet(; time=tend, safety_factor=0.7)
        job = Peridynamics.Job(body, vv; path, freq, fields=(:displacement, :velocity))
        Peridynamics.submit(job; quiet=true)
        write(kennung_datei, kennung)
    end
    return lese_stab_3d(vtk_path, label; l, Δx, rho)
end

# u_x und v_x je x-Spalte über den Querschnitt mitteln
function lese_stab_3d(vtk_path, label; l, Δx, rho)
    files = sort(filter(f -> endswith(f, ".vtu") || endswith(f, ".pvtu"),
                        readdir(vtk_path; join=true)))
    r0 = Peridynamics.read_vtk(files[1])
    x0 = r0[:position][1, :] .- r0[:displacement][1, :] .+ l / 2   # Referenzlage
    spalte = round.(Int, (x0 .- Δx / 2) ./ Δx) .+ 1
    nx = maximum(spalte)
    anzahl = zeros(nx)
    for s in spalte
        anzahl[s] += 1
    end
    mittel(f) = (m = zeros(nx); for (k, s) in enumerate(spalte); m[s] += f[k]; end; m ./ anzahl)
    time = Float64[]
    U = Vector{Float64}[]
    V = Vector{Float64}[]
    for f in files
        r = Peridynamics.read_vtk(f)
        push!(time, only(r[:time]))
        push!(U, mittel(r[:displacement][1, :]))
        push!(V, mittel(r[:velocity][1, :]))
    end
    p = sortperm(time)   # Dateinamen sind nicht einheitlich nummeriert -> nach Zeit sortieren
    x = [(s - 0.5) * Δx for s in 1:nx]
    hist = (; time=time[p], position=x, displacement=reduce(hcat, U[p]),
            velocity=reduce(hcat, V[p]), b_int=zeros(nx, length(time)))
    return (; label, hist, volume=fill(Δx, nx), rho)
end
