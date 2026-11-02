# Bild zur Dispersion im Stab (Kontinuum): links, was eine Bindung der Länge ξ von einer langen
# und einer kurzen Welle sieht, rechts die Phasengeschwindigkeit ω/(c_L k) über kδ für BB, CC, BAQP.
#
#   julia --project=scripts/vergleich scripts/vergleich/bild_dispersion_stab.jl [ausgabepfad]

using Printf
include("plotting.jl")
include("balken_analytisch.jl")   # integralsinus

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "stab")

"Cin(a) = ∫₀^a (1 - cos t)/t dt (Simpson)"
function cin(a; n=2000)
    t = range(0, a; length=2n + 1)
    f = [s == 0 ? 0.0 : (1 - cos(s)) / s for s in t]
    return step(t) / 3 * (f[1] + f[end] + 4sum(f[2:2:end-1]) + 2sum(f[3:2:end-2]))
end

# ω/(c_L k) im Kontinuum, a = kδ
phase_bb(a) = sqrt(4cin(a)) / a                                  # ρω² = (4E/δ²) Cin(kδ)
phase_cc(a) = abs(3 * (sin(a) - a * cos(a)) / a^3)              # ρω² = E (S/K)²
phase_baqp(a) = sqrt(2 * (a * integralsinus(a) - (1 - cos(a)))) / a   # ρω² = (E/δ) ∫ (1 - cos kξ)/ξ² dξ

fig = with_theme(THEMA) do
    f = Figure(size=(900, 340))
    # links: eine Bindung (x, x + ξ) vor einer langen und einer kurzen Welle
    gl = f[1, 1] = GridLayout()
    ξ = 1.0
    for (zeile, (λ, titel)) in enumerate(((6ξ, "lange Welle, λ = 6ξ: Bindung gedehnt"),
                                          (ξ, "kurze Welle, λ = ξ: Bindung nicht gedehnt")))
        ax = Axis(gl[zeile, 1]; title=titel, titlesize=13, xlabel=zeile == 2 ? L"x" : "",
                  ylabel=L"u", yticksvisible=false, yticklabelsvisible=false,
                  xticksvisible=false, xticklabelsvisible=false,
                  xgridvisible=false, ygridvisible=false)
        # Welle mit der steilsten Flanke bei x = 1, Bindung symmetrisch dazu
        welle(x) = sin(2π * (x - 1) / λ)
        x0 = 1 - ξ / 2
        xs = range(-0.5, 2.5; length=600)
        lines!(ax, xs, welle.(xs); color=REF_FARBE, linewidth=1.5)
        u0, u1 = welle(x0), welle(x0 + ξ)
        lines!(ax, [x0, x0 + ξ], [-1.35, -1.35]; color=FARBEN["BB"], linewidth=3)
        text!(ax, x0 + ξ / 2, -1.35; text=L"\xi", align=(:center, :bottom), fontsize=14)
        for (xp, up) in ((x0, u0), (x0 + ξ, u1))
            lines!(ax, [xp, xp], [-1.35, up]; color=(FARBEN["BB"], 0.5), linestyle=:dot)
        end
        scatter!(ax, [x0, x0 + ξ], [u0, u1]; color=FARBEN["BB"], markersize=10)
        xd = x0 + ξ + 0.12
        lines!(ax, [x0, xd], [u0, u0]; color=(FARBEN["CC"], 0.6), linestyle=:dot)
        if abs(u1 - u0) > 1e-6
            lines!(ax, [xd, xd], [u0, u1]; color=FARBEN["CC"], linewidth=2)
            text!(ax, xd + 0.05, (u0 + u1) / 2; text=L"\Delta u", align=(:left, :center),
                  fontsize=14, color=FARBEN["CC"])
        else
            text!(ax, xd + 0.05, u0; text=L"\Delta u = 0", align=(:left, :center),
                  fontsize=14, color=FARBEN["CC"])
        end
        limits!(ax, -0.5, 2.5, -1.6, 1.25)
    end
    # rechts: Phasengeschwindigkeit
    a = range(0.01, 10; length=500)
    ax = Axis(f[1, 2]; xlabel=L"k\delta", ylabel=L"\omega/(c_L k)",
              title="Phasengeschwindigkeit im Kontinuum")
    hlines!(ax, 1.0; color=REF_FARBE, linestyle=:dash, linewidth=1.5, label="klassisch")
    lines!(ax, a, phase_baqp.(a); color=FARBEN["BAQP"], label="BAQP")
    lines!(ax, a, phase_bb.(a); color=FARBEN["BB"], label="BB")
    lines!(ax, a, phase_cc.(a); color=FARBEN["CC"], label="CC")
    limits!(ax, 0, 10, 0, 1.1)
    axislegend(ax; position=:lb)
    colsize!(f.layout, 1, Relative(0.5))
    f
end
speichern(fig, joinpath(OUT, "00_dispersion_kontinuum"))
@printf("ω/(c_L k) bei kδ = 1: BB %.4f (1 - 1/48 = %.4f), CC %.4f (1 - 1/10), BAQP %.4f (1 - 1/72)\n",
        phase_bb(1.0), 1 - 1 / 48, phase_cc(1.0), phase_baqp(1.0))
println("Bild in ", abspath(OUT))
