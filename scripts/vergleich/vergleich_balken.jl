# Vergleich der peridynamischen Bernoulli-Balken (CC: gemittelte Krümmung, CC + RK: mit
# Reproducing-Kernel-Randkorrektur, BP: Mittel der Quadrate der Paarkrümmung) mit der
# klassischen Balkentheorie: Dispersion, Eigenfrequenzen, Biegelinie, losgelassener Kragarm.
#
#   julia --project=scripts/vergleich scripts/vergleich/vergleich_balken.jl [ausgabepfad]

using BBVV1D, Printf
include("plotting.jl")
include("balken_analytisch.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "balken")

# PMMA-Rundstab wie am Hopkinson-Bar, SI-Einheiten
const E = 3.0e9            # Pa
const RHO = 1180.0         # kg/m³
const D = 0.02             # m
const EI = E * π * D^4 / 64
const RHOA = RHO * π * D^2 / 4
const CB = sqrt(EI / RHOA)  # m²/s
const LBALKEN = 2.0        # m
const LKRAGARM = 1.0       # m, losgelassener Kragarm
const MFAK = 3.015

const MODELLE = ["CC" => MittlereKruemmung(), "CC + RK" => MittlereKruemmung(; randkorrektur=true),
                 "BP" => BindungsKruemmung()]

system(m, l, N; mfak=MFAK, kw...) = BeamSystem(l, N, mfak * l / N, EI, RHOA, m; kw...)
maxfehler(run, ref) = maximum(fehler(run, ref))

printstyled(@sprintf("PMMA-Balken d = %.0f mm: EI = %.3f N m², ρA = %.4f kg/m, c_B = %.3f m²/s\n",
                     1e3D, EI, RHOA, CB); color=:blue, bold=true)

##-- Dispersion ------------------------------------------------------------------------
kΔx = range(1e-3, π; length=400)
Δx0 = LBALKEN / 200
dispB(m) = dispersion_balken.(m, kΔx ./ Δx0; Δx=Δx0, δ=MFAK * Δx0)
speichern(plot_dispersion([("CC, CC + RK", kΔx, dispB(:A), FARBEN["CC"]),
                           ("BP", kΔx, dispB(:B), FARBEN["BP"])];
                          title=L"\delta = 3.015\,\Delta x", ylabel=L"\omega/(c_B\,k^2)"),
          joinpath(OUT, "01_dispersion"))

##-- Eigenfrequenzen (l = 2 m, N = 200) --------------------------------------------------
N0 = 200
nmod = 40
zk = kragarm_wurzeln(nmod)
exakt = Dict(:gelenkig => [(n * π / LBALKEN)^2 * CB for n in 1:nmod],
             :kragarm => (zk ./ LBALKEN) .^ 2 .* CB)
lager = Dict(:gelenkig => (; links=:gelenkig, rechts=:gelenkig),
             :kragarm => (; links=:eingespannt, rechts=:frei))
verh = Dict{Tuple{Symbol,String},Vector{Float64}}()
for fall in (:gelenkig, :kragarm), (label, m) in MODELLE
    ω, _ = eigenfrequenzen(system(m, LBALKEN, N0; lager[fall]...))
    verh[(fall, label)] = ω[1:nmod] ./ exakt[fall]
end
for (fall, titel) in ((:gelenkig, "gelenkig gelagert"), (:kragarm, "Kragarm"))
    speichern(plot_study([(label, collect(1:nmod), verh[(fall, label)]) for (label, _) in MODELLE];
                         xlabel=L"\mathrm{Mode}\ n", ylabel=L"\omega_n/\omega_n^\mathrm{exakt}",
                         xscale=identity, yscale=identity, title=titel * ", N = $N0",
                         legendpos=:rt),
              joinpath(OUT, fall === :gelenkig ? "02a_eigenfrequenzen_gelenkig" :
                            "02b_eigenfrequenzen_kragarm"))
end
println("Eigenfrequenzen ω_n/ω_exakt (N = $N0), n = 1, 2, 5, 10, 20:")
for fall in (:gelenkig, :kragarm), (label, _) in MODELLE
    v = verh[(fall, label)]
    @printf("  %-9s %-7s %s\n", fall, label, join((@sprintf("%.4f", v[n]) for n in (1, 2, 5, 10, 20)), "  "))
end

##-- Statik: Kragarm mit Endlast F, gelenkig mit Streckenlast q ----------------------------
F, q = 1.0, 1.0
kragarm_ref(x) = F * x^2 * (3LBALKEN - x) / (6EI)
gelenkig_ref(x) = q * x * (LBALKEN^3 - 2LBALKEN * x^2 + x^3) / (24EI)
function statik_fehler(m, N; mfak=MFAK)
    s = system(m, LBALKEN, N; mfak)
    f = zeros(length(s.dofs)); f[end] = F
    wk = knotenwerte(s, statik(s, f))
    s2 = system(m, LBALKEN, N; mfak, links=:gelenkig, rechts=:gelenkig)
    wg = knotenwerte(s2, statik(s2, q .* s2.volume[s2.dofs .+ 1]))
    return (; kragarm=abs(wk[end] / kragarm_ref(LBALKEN) - 1),
            gelenkig=abs(wg[N ÷ 2 + 1] / gelenkig_ref(LBALKEN / 2) - 1), x=s.x, wk)
end
# Biegelinie bei grober Diskretisierung
Ngrob = 40
fig = with_theme(THEMA) do
    f = Figure(size=(620, 320))
    ax = Axis(f[1, 1]; xlabel=L"x\;[\mathrm{m}]", ylabel=L"w\;[\mathrm{m}]",
              title="Kragarm, Endlast F = 1 N, N = $Ngrob")
    xr = range(0, LBALKEN; length=201)
    lines!(ax, xr, kragarm_ref.(xr); color=REF_FARBE, linestyle=:dash, linewidth=1.5,
           label=REF_LABEL)
    for (i, (label, m)) in enumerate(MODELLE)
        r = statik_fehler(m, Ngrob)
        scatterlines!(ax, r.x, r.wk; color=farbe(label, i), markersize=6, label)
    end
    axislegend(ax; position=:lt)
    f
end
speichern(fig, joinpath(OUT, "03a_biegelinie_kragarm"))
Ns = [25, 50, 100, 200, 400, 800]
fs = Dict(label => [statik_fehler(m, N) for N in Ns] for (label, m) in MODELLE)
for (art, titel) in ((:kragarm, "Kragarm, Durchbiegung am Ende"),
                     (:gelenkig, "gelenkig, Durchbiegung in der Mitte"))
    speichern(plot_study([(label, LBALKEN ./ Ns, [getfield(r, art) for r in fs[label]])
                          for (label, _) in MODELLE];
                         xlabel=L"\Delta x", ylabel="relativer Fehler",
                         xticks=(LBALKEN ./ Ns, ["l/$N" for N in Ns]),
                         title=titel * ", δ = 3.015 Δx", legendpos=:rb),
              joinpath(OUT, art === :kragarm ? "03b_konvergenz_kragarm" : "03c_konvergenz_gelenkig"))
end
mfaks = [1.015, 2.015, 3.015, 4.015, 6.015]
fm = Dict(label => [statik_fehler(m, 200; mfak=mf) for mf in mfaks] for (label, m) in MODELLE)
speichern(plot_study([(label, mfaks, [r.kragarm for r in fm[label]]) for (label, _) in MODELLE];
                     xlabel=L"\delta/\Delta x", ylabel="relativer Fehler", xscale=identity,
                     xticks=mfaks, title=L"\mathrm{Kragarm},\ N = 200", legendpos=:rb),
          joinpath(OUT, "03d_horizont_kragarm"))
println("Statik, relativer Fehler Kragarm (Ende) | gelenkig (Mitte), N = $(Ns):")
for (label, _) in MODELLE
    @printf("  %-7s %s | %s\n", label, join((@sprintf("%.1e", r.kragarm) for r in fs[label]), " "),
            join((@sprintf("%.1e", r.gelenkig) for r in fs[label]), " "))
end

# Verifikation des CC-Modells gegen die Kontinuumslösung (Yang u. a. 2022) bei festem δ:
# Geisterknoten und Faltungsoperator sind beim gelenkigen Balken gleichwertig, die Abweichung ist
# nur der Quadraturfehler der Horizontsumme
δrs = [0.05, 0.1, 0.2]
Nys = [50, 100, 200, 400, 800, 1600]
yang_daten = map(δrs) do δr
    δ = δr * LBALKEN
    wref(x) = yang_gelenkig(x; l=LBALKEN, δ, q, EI)
    e = map(Nys) do N
        s = BeamSystem(LBALKEN, N, δ * (1 + 1e-9), EI, RHOA, MittlereKruemmung();
                       links=:gelenkig, rechts=:gelenkig)
        w = knotenwerte(s, statik(s, q .* s.volume[s.dofs .+ 1]))
        maximum(abs(w[i + 1] - wref(s.x[i + 1])) for i in 0:N÷10:N) / wref(LBALKEN / 2)
    end
    @printf("CC gegen Yang u. a., δ = %.2f l: w(l/2)/w_klass = %.5f, Fehler %s\n", δr,
            wref(LBALKEN / 2) / gelenkig_ref(LBALKEN / 2), join((@sprintf("%.1e", v) for v in e), " "))
    (@sprintf("δ = %.2f l", δr), LBALKEN ./ Nys, e)
end
speichern(plot_study(yang_daten; xlabel=L"\Delta x", ylabel="relativer Fehler",
                     xticks=(LBALKEN ./ Nys, ["l/$N" for N in Nys]),
                     title="CC gegen Kontinuumslösung, gelenkig", legendpos=:rb),
          joinpath(OUT, "03e_verifikation_yang"))

##-- Dynamik: losgelassener Kragarm (l = 1 m) ---------------------------------------------
κ0 = 0.01                                   # 1/m, Endauslenkung κ₀l²/2 = 5 mm
T1 = 2π / ((kragarm_wurzeln(1)[1] / LKRAGARM)^2 * CB)
ref = kragarm_loslassen(; l=LKRAGARM, c=CB, κ0, n_moden=40)
@printf("Kragarm l = %.1f m: T₁ = %.4f s, Modalreihe bei t = 0: max|w - w₀| = %.1e m\n",
        LKRAGARM, T1, maximum(abs(ref(x, 0.0) - κ0 * x^2 / 2) for x in range(0, LKRAGARM; length=101)))
Ndyn = 200
runs_dyn = map(MODELLE) do (label, m)
    s = system(m, LKRAGARM, Ndyn)
    x = s.x[s.dofs .+ 1]
    hist = balken_dynamik(s; w0=κ0 .* x .^ 2 ./ 2, tend=2T1, n_out=400)
    (; label, hist, volume=s.volume, rho=RHOA)
end
speichern(plot_probe(runs_dyn, [LKRAGARM / 2, LKRAGARM]; ref, tnorm=T1, tlabel=L"t/T_1",
                     ylabel=L"w\;[\mathrm{m}]"),
          joinpath(OUT, "04a_kragarm_messpunkte"))
speichern(plot_snapshots(runs_dyn, [0.1, 0.25, 0.5, 1.0, 1.5] .* T1; ref, tnorm=T1,
                         tlabel="t/T₁", ylabel=L"w\;[\mathrm{m}]"),
          joinpath(OUT, "04b_kragarm_momentaufnahmen"))
speichern(plot_error(runs_dyn, ref; tnorm=T1, tlabel=L"t/T_1"), joinpath(OUT, "04c_kragarm_fehler"))
println("Kragarm losgelassen, max. relativer Fehler (N = $Ndyn, 2 T₁):")
for r in runs_dyn
    @printf("  %-7s %.4f\n", r.label, maxfehler(r, ref))
end
println("\nDiagramme in ", abspath(OUT))
