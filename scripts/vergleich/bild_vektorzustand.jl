# Bild zu Vektorzustand und gemitteltem Deformationsgradienten (Kap65, Abb. FIG:States):
# links die Familie in der Referenzkonfiguration, Mitte ein allgemeiner Vektorzustand Δx mit seiner
# besten linearen Näherung F̄ (Ellipse), rechts eine Nullenergiemode mit F̄ = I.
#
#   julia --project=scripts/vergleich scripts/vergleich/bild_vektorzustand.jl [ausgabepfad]

using Printf, LinearAlgebra
include("plotting.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "kap65", "states_neu")

# Familie: zwei Ringe um X^k = 0, Kernel w = 1, Volumen V = 1
ring(n, r, φ0) = [r .* [cos(φ0 + 2π * j / n), sin(φ0 + 2π * j / n)] for j in 0:n-1]
const AUSSEN = ring(12, 1.0, 0.0)
const INNEN = ring(6, 0.5, π / 6)
const XI = vcat(AUSSEN, INNEN)

"gemittelter Deformationsgradient F̄ = (Σ Δx ⊗ ξ) K⁻¹ mit K = Σ ξ ⊗ ξ"
function fbar(dx)
    K = sum(ξ * ξ' for ξ in XI)
    return sum(d * ξ' for (d, ξ) in zip(dx, XI)) / K
end

# allgemeiner Zustand: nichtlineare Abbildung und eine aufgerissene Ecke (nicht glatt)
function zustand(ξ)
    y = [1.15ξ[1] + 0.1ξ[2] + 0.18ξ[2]^2, 0.85ξ[2] + 0.22ξ[1]^2]
    ξ[1] > 0.2 && ξ[2] < -0.2 && (y += [0.22, -0.28])
    return y
end
const DX_ALLG = zustand.(XI)
DX_ALLG[5] *= 0.5   # ein Randpunkt weit nach innen: jede Bindung wird für sich abgebildet
# Nullenergiemode: radial abwechselnd nach außen und innen, Σ Δu ⊗ ξ = 0, also F̄ = I
const DX_NULL = vcat([(1 + 0.18 * (-1)^j) * ξ for (j, ξ) in enumerate(AUSSEN)],
                     [(1 + 0.25 * (-1)^j) * ξ for (j, ξ) in enumerate(INNEN)])

const C_REF = "#9a9a96"
const C_ZUST = FARBEN["BB"]
const C_FBAR = FARBEN["CC"]
const C_REST = FARBEN["BAQP"]

geschlossen(ps) = (xs = first.(ps); ys = last.(ps); (vcat(xs, xs[1]), vcat(ys, ys[1])))

function familie!(ax, dx; F=nothing, rest=false)
    for d in dx
        lines!(ax, [0, d[1]], [0, d[2]]; color=(C_ZUST, 0.45), linewidth=1)
    end
    lines!(ax, geschlossen(dx[1:12])...; color=C_ZUST, linewidth=1.8)
    if F !== nothing
        θ = range(0, 2π; length=200)
        e = [F * [cos(t), sin(t)] for t in θ]
        lines!(ax, first.(e), last.(e); color=C_FBAR, linewidth=2, linestyle=:dash)
        fx = [F * ξ for ξ in XI]
        if rest
            for (p, d) in zip(fx, dx)
                lines!(ax, [p[1], d[1]], [p[2], d[2]]; color=C_REST, linewidth=2)
            end
        end
        scatter!(ax, first.(fx), last.(fx); color=:white, strokecolor=C_FBAR, strokewidth=1.5,
                 markersize=7)
    end
    scatter!(ax, first.(dx), last.(dx); color=C_ZUST, markersize=8)
    scatter!(ax, [0.0], [0.0]; color=:black, markersize=9)
end

fig = with_theme(THEMA) do
    f = Figure(size=(900, 310))
    achse(i, titel) = Axis(f[1, i]; title=titel, titlesize=19, aspect=DataAspect(),
                           xgridvisible=false, ygridvisible=false, leftspinevisible=false,
                           bottomspinevisible=false, xticksvisible=false, yticksvisible=false,
                           xticklabelsvisible=false, yticklabelsvisible=false)
    # Referenz
    ax = achse(1, L"Referenz")
    θ = range(0, 2π; length=200)
    lines!(ax, cos.(θ), sin.(θ); color=C_REF, linestyle=:dot, linewidth=1.5)
    for ξ in XI
        lines!(ax, [0, ξ[1]], [0, ξ[2]]; color=(C_REF, 0.8), linewidth=1)
    end
    scatter!(ax, first.(XI), last.(XI); color=C_REF, markersize=8)
    scatter!(ax, [0.0], [0.0]; color=:black, markersize=9)
    text!(ax, 0.05, -0.12; text=L"\mathbf{X}^k", fontsize=21, align=(:left, :top))
    text!(ax, AUSSEN[2]...; text=L"\xi", fontsize=21, align=(:left, :bottom),
          offset=(4, 2))
    # allgemeiner Zustand
    F = fbar(DX_ALLG)
    ax = achse(2, L"allgemeiner Vektorzustand")
    familie!(ax, DX_ALLG; F, rest=true)
    text!(ax, 0.05, -0.12; text=L"\mathbf{x}^k", fontsize=21, align=(:left, :top))
    # Nullenergiemode
    F0 = fbar(DX_NULL)
    ax = achse(3, L"Nullenergiemode, $\bar{\mathbf{F}} = \mathbf{I}$")
    familie!(ax, DX_NULL; F=F0, rest=true)
    for ax in filter(c -> c isa Axis, f.content)
        limits!(ax, -1.45, 1.45, -1.45, 1.3)
    end
    # Legende
    elem = [LineElement(color=C_ZUST, linewidth=1.8), LineElement(color=C_FBAR, linestyle=:dash),
            LineElement(color=C_REST, linewidth=2)]
    Legend(f[2, 1:3], elem, [L"Vektorzustand $\Delta\mathbf{x}$ (je Bindung beliebig)",
                             L"Bild des Kreises unter $\bar{\mathbf{F}}$ (Ellipse), Punkte $\bar{\mathbf{F}}\,\xi$",
                             L"Rest $\Delta\mathbf{x} - \bar{\mathbf{F}}\,\xi$: von $\bar{\mathbf{F}}$ nicht erfasst"];
           orientation=:horizontal, tellheight=true, framevisible=false, labelsize=17,
           nbanks=2)
    rowgap!(f.layout, 4)
    f
end
speichern(fig, OUT)
F = fbar(DX_ALLG)
@printf("F̄ allgemein = [%.3f %.3f; %.3f %.3f], Nullenergiemode |F̄ - I| = %.1e\n",
        F[1, 1], F[1, 2], F[2, 1], F[2, 2], norm(fbar(DX_NULL) - I))
println("Bild in ", abspath(OUT))
