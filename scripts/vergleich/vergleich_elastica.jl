# Peridynamische Elastica (Modell B mit Winkelkrümmung + BB-Dehnung) gegen die exakte Elastica
# des gelenkig gelagerten Druckstabes: Last-Verkürzung, Durchbiegung, Gestalt, Konvergenz.
#
#   julia --project=scripts/vergleich scripts/vergleich/vergleich_elastica.jl [ausgabepfad]

using BBVV1D, Printf
include("plotting.jl")
include("elastica_analytisch.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "elastica")

# Knickstab wie in knickstab.jl (zentrisch gedrückt), SI-Einheiten
const E = 3.375e9           # Pa
const D = 1.66e-3           # m
const L = 0.254             # m, Stablänge zwischen den Gelenken
const EI = E * π * D^4 / 64
const EA = E * π * D^2 / 4
const FCR = π^2 * EI / L^2
const MFAK = 3.015
const QIMP = 1e-5 * FCR     # Querkraft in der Mitte als Imperfektion

printstyled(@sprintf("Knickstab: L = %.0f mm, d = %.2f mm, EI = %.4e N m², F_cr = %.4f N\n",
                     1e3L, 1e3D, EI, FCR); color=:blue, bold=true)

# Verkürzungen: ab kurz nach der Verzweigung gleichmäßig bis zur Schleife (Sehne < 0)
const DELTAS = vcat([1e-3, 3e-3], range(0.01, 1.5; length=60)) .* L

function rechne_pfad(N; mfak=MFAK)
    sys = ElasticaSystem(L, N, mfak * L / N, EI, EA)
    return sys, elastica_pfad(sys, DELTAS; Q=QIMP)
end

wmax(r) = maximum(abs, r.Y[2, :])

N0 = 100
sys, pfad = rechne_pfad(N0)
ex = [elastica_zur_sehne(1 - r.Δ / L) for r in pfad]

##-- Last-Verkürzung und Durchbiegung ---------------------------------------------------------
pfad_exakt = [elastica_zur_sehne(c) for c in range(0.9999, -0.5; length=300)]
fig = with_theme(THEMA) do
    f = Figure(size=(820, 340))
    ax1 = Axis(f[1, 1]; xlabel=L"\Delta/L", ylabel=L"F/F_\mathrm{cr}", title="Last-Verkürzung")
    lines!(ax1, [1 - c for c in range(0.9999, -0.5; length=300)], [e.F_Fcr for e in pfad_exakt];
           color=REF_FARBE, linestyle=:dash, linewidth=1.5, label="Elastica exakt")
    scatter!(ax1, [r.Δ / L for r in pfad], [r.F / FCR for r in pfad]; color=FARBEN["B"],
             markersize=7, label="PD, N = $N0")
    axislegend(ax1; position=:lt)
    ax2 = Axis(f[1, 2]; xlabel=L"F/F_\mathrm{cr}", ylabel=L"w_\mathrm{max}/L",
               title="Durchbiegung in der Mitte")
    lines!(ax2, [e.F_Fcr for e in pfad_exakt], [e.wmax_L for e in pfad_exakt]; color=REF_FARBE,
           linestyle=:dash, linewidth=1.5, label="Elastica exakt")
    scatter!(ax2, [r.F / FCR for r in pfad], [wmax(r) / L for r in pfad]; color=FARBEN["B"],
             markersize=7, label="PD, N = $N0")
    axislegend(ax2; position=:rb)
    f
end
speichern(fig, joinpath(OUT, "01_last_verkuerzung"))

##-- Gestalt ------------------------------------------------------------------------------
zeige = [0.2, 0.5, 0.8, 1.1, 1.4]                       # Δ/L
fig = with_theme(THEMA) do
    f = Figure(size=(620, 420))
    ax = Axis(f[1, 1]; xlabel=L"x/L", ylabel=L"z/L", aspect=DataAspect(),
              title="Gestalt bei Δ/L = " * join(string.(zeige), ", "))
    for (i, Δr) in enumerate(zeige)
        r = pfad[argmin(abs.([p.Δ / L for p in pfad] .- Δr))]
        e = elastica_zur_sehne(1 - r.Δ / L)
        xe, we = elastica_gestalt(e.p, L)
        lines!(ax, xe ./ L, sign(r.Y[2, end ÷ 2]) .* we ./ L; color=REF_FARBE, linestyle=:dash,
               linewidth=1.5, label=i == 1 ? "Elastica exakt" : nothing)
        reale = (sys.nmax + 1):(sys.nmax + 1 + sys.N)
        scatter!(ax, r.Y[1, reale] ./ L, r.Y[2, reale] ./ L; color=FARBEN["B"], markersize=4,
                 label=i == 1 ? "PD, N = $N0" : nothing)
    end
    axislegend(ax; position=:rt)
    f
end
speichern(fig, joinpath(OUT, "02_gestalt"))

##-- Fehler entlang des Pfades und Konvergenz -------------------------------------------------
println("Δ/L     F/F_cr PD   exakt     Fehler F    w_max/L PD  exakt    Fehler w")
for (r, e) in zip(pfad, ex)
    r.Δ / L < 0.05 && continue
    i = findfirst(==(r), pfad)
    i % 6 == 0 || continue
    @printf("%5.3f   %8.5f  %8.5f  %9.2e   %8.5f  %8.5f  %9.2e\n", r.Δ / L, r.F / FCR, e.F_Fcr,
            r.F / FCR / e.F_Fcr - 1, wmax(r) / L, e.wmax_L, wmax(r) / L / e.wmax_L - 1)
end
Ns = [25, 50, 100, 200, 400]
ziele = [0.5, 1.0]                                      # Δ/L für die Konvergenzstudie
fehler_N = Dict(z => Float64[] for z in ziele)
for N in Ns
    s, p = rechne_pfad(N)
    for z in ziele
        r = p[argmin(abs.([x.Δ / L for x in p] .- z))]
        push!(fehler_N[z], abs(r.F / FCR / elastica_zur_sehne(1 - r.Δ / L).F_Fcr - 1))
    end
end
speichern(plot_study([(@sprintf("Δ/L = %.1f", z), L ./ Ns, fehler_N[z], c)
                      for (z, c) in zip(ziele, [FARBEN["B"], FARBEN["A"]])];
                     xlabel=L"\Delta s", ylabel=L"|F/F_\mathrm{exakt} - 1|",
                     xticks=(L ./ Ns, ["L/$N" for N in Ns]), title="δ = 3.015 Δs",
                     legendpos=:rb),
          joinpath(OUT, "03_konvergenz"))
println("Konvergenz |F/F_exakt - 1|, N = $Ns:")
for z in ziele
    println("  Δ/L = $z: ", join((@sprintf("%.2e", e) for e in fehler_N[z]), "  "))
end
println("\nDiagramme in ", abspath(OUT))
