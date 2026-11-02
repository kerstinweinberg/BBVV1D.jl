# Exzentrisch gedrückter Stab (Spaghetti, Weinberg: Breaking Spaghetti, Tab. 1 und A2): links
# gelenkig, rechts greift die Kraft F an einem starren Hebel der Länge e = 0.01ℓ an, senkrecht zum
# Stabende. Kraftsteuerung bis zum Bruch (Randspannung σ_F am Scheitel), BP- und CC-Elastica.
#
#   julia --project=scripts/vergleich scripts/vergleich/vergleich_exzentrisch.jl [ausgabepfad]

using BBVV1D, Printf
include("plotting.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "elastica")

# Tab. 1, SI-Einheiten
const L = 0.254             # m, Stablänge 2ℓ
const D = 1.66e-3           # m
const E = 3.375e9           # Pa
const SIGF = 30e6           # Pa
const EI = E * π * D^4 / 64
const EA = E * π * D^2 / 4
const AQ = π * D^2 / 4
const WQ = π * D^3 / 32     # Widerstandsmoment
const FCR = π^2 * EI / L^2
const EXZ = 0.01 * L / 2    # e = 0.01ℓ
const N = 200
const MFAK = 3.015

# Tab. A2: F/Fcr, s_c [mm], w_max [mm], w(2ℓ) [mm], ℓ_ch [mm], κ_max [1/m]
const REF = [(0.25, 254.0, 1.270, 1.2700, 253.996, 0.049), (0.50, 179.6, 1.596, 1.2700, 253.990, 0.122),
             (0.80, 142.0, 3.894, 1.2689, 253.895, 0.477), (0.90, 133.9, 7.808, 1.2648, 253.495, 1.075),
             (0.95, 130.6, 14.66, 1.2503, 252.069, 2.131), (1.00, 128.4, 33.77, 1.1594, 242.985, 5.166),
             (1.10, 127.5, 66.99, 0.7905, 204.979, 11.27)]

printstyled(@sprintf("Exzentrischer Stab: 2ℓ = %.0f mm, d = %.2f mm, EI = %.4e N m², F_cr = %.5f N, e = %.2f mm\n",
                     1e3L, 1e3D, EI, FCR, 1e3EXZ); color=:blue, bold=true)

"Scheitel (größtes z, parabolisch zwischen den Knoten), Bogenlänge dort, Endlage, Sehne, κ_max"
function auswerten(sys, r)
    Y = r.Y[:, (sys.nmax + 1):(sys.nmax + 1 + sys.N)]   # reale Knoten 0..N
    z = Y[2, :]
    k = argmax(z)
    sc, wmax = (k - 1) * sys.Δs, z[k]
    if 1 < k < length(z)                                  # Parabel durch drei Punkte
        a, b, c = z[k-1], z[k], z[k+1]
        t = 0.5 * (a - c) / (a - 2b + c)
        sc += t * sys.Δs
        wmax = b - 0.25 * (a - c) * t
    end
    κ = abs.(elastica_kruemmung(sys, r.Y))
    return (; sc, wmax, wend=z[end], lch=Y[1, end], κmax=maximum(κ), κ,
            U=elastica_energie(sys, r.Y))
end

randspannung(F, wmax) = -F / AQ + F * wmax / WQ

ergebnisse = Dict{Symbol,Any}()
for modell in (:BP, :CC)
    sys = ElasticaSystem(L, N, MFAK * L / N, EI, EA; rechts_frei=true, modell)
    fak = vcat(0.25, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0, 1.02:0.02:1.1)
    pfad = exzentrisch_pfad(sys, fak .* FCR; e=EXZ)
    printstyled("\n== $modell, N = $N, δ = $MFAK Δs\n"; bold=true)
    println("F/Fcr   s_c [mm]  (ref)    w_max [mm] (ref)     w(2ℓ) [mm] (ref)    ℓ_ch [mm] (ref)      κ_max [1/m] (ref)")
    for (f0, sc0, w0, we0, l0, κ0) in REF
        r = pfad[findfirst(≈(f0), fak)]
        a = auswerten(sys, r)
        @printf("%.2f  %7.1f (%5.1f)  %7.3f (%6.3f)  %7.4f (%6.4f)  %8.3f (%7.3f)  %7.3f (%6.3f)\n", f0,
                1e3a.sc, sc0, 1e3a.wmax, w0, 1e3a.wend, we0, 1e3a.lch, l0, a.κmax, κ0)
    end
    # Bruch: Randspannung -F/A + F w_max/W = σ_F (Sekantenverfahren in F)
    g(r) = randspannung(r.F, auswerten(sys, r).wmax) - SIGF
    r1, r2 = pfad[end-1], pfad[end]
    for _ in 1:8
        F3 = r2.F - g(r2) * (r2.F - r1.F) / (g(r2) - g(r1))
        r3 = only(exzentrisch_pfad(sys, [F3]; e=EXZ, q0=r2.q))
        r1, r2 = r2, r3
        abs(g(r2)) < 1e-6 * SIGF && break
    end
    a = auswerten(sys, r2)
    @printf("Bruch: F = %.4f Fcr = %.4f N (1.0891, 0.2096), w_max = %.2f mm (64.49), ℓ_ch = %.2f mm (209.14),\n",
            r2.F / FCR, r2.F, 1e3a.wmax, 1e3a.lch)
    @printf("       s_c = %.1f mm (127.5), κ_max = %.2f 1/m (10.74), M = %.2f N mm (13.52), U = %.2f N mm (9.04)\n",
            1e3a.sc, a.κmax, 1e3 * r2.F * a.wmax, 1e3a.U)
    ergebnisse[modell] = (; sys, pfad, fak, bruch=r2, a)
end

# Bild: Biegelinien wie Abb. 2 und Krümmung im Bruchzustand gegen κ = k²w (aus M = F w)
fig = with_theme(THEMA) do
    f = Figure(size=(900, 330))
    ax = Axis(f[1, 1]; xlabel=L"x\ \mathrm{[mm]}", ylabel=L"w\ \mathrm{[mm]}",
              title="Biegelinien, Punkte BP, Linie CC")
    for (i, f0) in enumerate((0.5, 0.8, 0.95, 1.0, 1.1))
        for (modell, stil) in ((:BP, :punkte), (:CC, :linie))
            e = ergebnisse[modell]
            r = e.pfad[findfirst(≈(f0), e.fak)]
            Y = r.Y[:, (e.sys.nmax + 1):(e.sys.nmax + 1 + e.sys.N)]
            farbe =(FARBEN["BB"], FARBEN["CC"], FARBEN["OSB"], FARBEN["CC |ξ|"], FARBEN["BAQP"])[i]
            if stil === :punkte
                scatter!(ax, 1e3 .* Y[1, 1:5:end], 1e3 .* Y[2, 1:5:end]; color=farbe, markersize=5)
            else
                lines!(ax, 1e3 .* Y[1, :], 1e3 .* Y[2, :]; color=farbe, linewidth=1.5,
                       label=Makie.LaTeXString(@sprintf("\$F/F_\\mathrm{cr} = %.2f\$", f0)))
            end
        end
    end
    axislegend(ax; position=:lt, labelsize=10)
    ax2 = Axis(f[1, 2]; xlabel=L"s\ \mathrm{[mm]}", ylabel=L"\kappa\ \mathrm{[1/m]}",
               title="Krümmung beim Bruch")
    for (modell, farbe) in ((:BP, FARBEN["BP"]), (:CC, FARBEN["CC"]))
        e = ergebnisse[modell]
        s = (0:e.sys.N) .* e.sys.Δs
        Y = e.bruch.Y[:, (e.sys.nmax + 1):(e.sys.nmax + 1 + e.sys.N)]
        lines!(ax2, 1e3 .* s, e.a.κ; color=farbe, label=String(modell))
        modell === :CC && lines!(ax2, 1e3 .* s, e.bruch.F / EI .* Y[2, :]; color=REF_FARBE,
                                 linestyle=:dash, label=L"k^2 w")
    end
    axislegend(ax2; position=:cb)
    f
end
speichern(fig, joinpath(OUT, "exzentrisch_01_biegelinie"))
println("Bild in ", abspath(OUT))
