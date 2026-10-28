# Einfluss der Querschnittsauflösung im 3D-Referenzstab (Peridynamics.jl):
# 500 × npyz × npyz Punkte, BB und OSB mit Energiekorrektur, ν = 0.25.
#
#   julia --project=scripts/vergleich -t auto scripts/vergleich/querschnitt_3d.jl
#
# Die Ergebnisse von npyz = 4 werden aus vergleich_stab.jl übernommen (gleicher Pfad).

using BBVV1D, Printf
include("analytisch.jl")
include("plotting.jl")
include("referenz_3d.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "stab")

const E = 3.2e9
const RHO = 1200.0
const C0 = sqrt(E / RHO)
const LSTAB = 1.0
const VMAX = 10.0
const TPULS = 0.1 * LSTAB / C0
const TEND = 2 * LSTAB / C0
const N3D = 500
const TNORM = LSTAB / C0

Δx = LSTAB / N3D
x0 = (floor(Int, 3.015Δx / Δx - 0.5) + 0.5) * Δx      # Innenkante der Randschicht
ref(x, t) = stab_analytisch(s -> puls_u(s, VMAX, TPULS), x, t; c=C0, x0, l=LSTAB)

# Wellengeschwindigkeit aus der Laufzeit des Pulsmaximums zwischen x = 0.3 l und 0.8 l
function wellengeschwindigkeit(run)
    h = run.hist
    tpk(x) = h.time[argmax(h.displacement[argmin(abs.(h.position .- x)), :])]
    return 0.5LSTAB / (tpk(0.8LSTAB) - tpk(0.3LSTAB))
end

kw3d = (; l=LSTAB, N=N3D, E, rho=RHO, vmax=VMAX, T=TPULS, tend=TEND)
materialien = ["BB" => Peridynamics.BBMaterial{Peridynamics.EnergySurfaceCorrection}(),
               "OSB" => Peridynamics.OSBMaterial{Peridynamics.EnergySurfaceCorrection}()]
ergebnisse = []
for npyz in (4, 8), (name, mat) in materialien
    sub = npyz == 4 ? lowercase(name) : lowercase(name) * "_$(npyz)x$(npyz)"
    t = @elapsed run = stab_3d(mat, "$name-3D $(npyz)×$(npyz)"; kw3d..., npyz,
                               path=joinpath(OUT, "3d", sub))
    c = wellengeschwindigkeit(run)
    e = maximum(fehler(run, ref))
    push!(ergebnisse, (; name, npyz, run, c, e))
    @printf("%-4s %d×%d: c/c₀ = %.4f   max. Fehler = %.4f   (%.0f s)\n", name, npyz, npyz,
            c / C0, e, t)
end

# Messpunkte: 4×4 gestrichelt, 8×8 durchgezogen
runs = [(; r.run..., label=r.run.label) for r in ergebnisse]
with_theme(THEMA) do
    fig = Figure(size=(620, 420))
    axs = Axis[]
    for (row, xp) in enumerate((0.5LSTAB, LSTAB))
        ax = Axis(fig[row, 1]; ylabel=L"u\;[\mathrm{m}]", title="x = $(xp) m",
                  titlealign=:left, titlesize=11)
        push!(axs, ax)
        t = ergebnisse[1].run.hist.time
        lines!(ax, t ./ TNORM, ref.(xp, t); color=REF_FARBE, linestyle=:dash,
               linewidth=1.5, label=REF_LABEL)
        for r in ergebnisse
            h = r.run.hist
            ip = argmin(abs.(h.position .- xp))
            lines!(ax, h.time ./ TNORM, h.displacement[ip, :];
                   color=FARBEN[r.name * "-3D"], linestyle=r.npyz == 4 ? :dot : :solid,
                   label=r.run.label)
        end
        xlims!(ax, xp == LSTAB ? (0.85, 1.25) : (0.4, 0.8))
        row == 1 && hidexdecorations!(ax; grid=false, ticklabels=false, ticks=false)
    end
    axs[end].xlabel = L"t\,c/l"
    Legend(fig[0, 1], axs[1]; orientation=:horizontal, nbanks=2)
    speichern(fig, joinpath(OUT, "12_querschnitt_3d"))
end

println("\nDiagramm in ", abspath(joinpath(OUT, "12_querschnitt_3d.pdf")))
