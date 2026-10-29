# Vergleich BB / CC / CC |ξ| / BAQP / OSB im 1D-Stab gegen die d'Alembert-Lösung und gegen
# 3D-Rechnungen mit Peridynamics.jl (github.com/kaipartmann/Peridynamics.jl).
# CC: gemittelter Deformationsgradient (CCFMaterial), CC |ξ|: ältere Variante (CCMaterial),
# BAQP: CC mit bindungsbezogener Integration (BAQPMaterial).
#
#   julia --project=scripts/vergleich -t auto scripts/vergleich/vergleich_stab.jl [ausgabepfad]
#
# Links Geschwindigkeitspuls (eine Sinusperiode, Pulslänge l/10) über eine Randschicht der
# Dicke δ, rechts freies Ende, Laufzeit 2 l/c. Δt wie in Peridynamics.jl aus der
# Bondkonstante bc, damit für alle Modelle gleich und die Ausgabezeitpunkte identisch.

using BBVV1D, Printf
include("analytisch.jl")
include("plotting.jl")
include("referenz_3d.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "stab")

# PMMA wie am Hopkinson-Bar, SI-Einheiten
const E = 3.0e9         # Pa
const RHO = 1180.0      # kg/m³
const C0 = sqrt(E / RHO)
const LSTAB = 2.0       # m
const VMAX = 10.0       # m/s
const TPULS = 0.1 * LSTAB / C0
const TEND = 2 * LSTAB / C0
const TNORM = LSTAB / C0
const TLABEL = L"t\,c/l"

# Standardmodelle: diskret kalibriert bzw. mit Energiekorrektur, OSB mit linearem Kernel
const MODELLE = ["BB" => BBMaterial{EnergySurfaceCorrection}(),
                 "CC" => CCFMaterial(),
                 "CC |ξ|" => CCMaterial(; calibration=:discrete),
                 "BAQP" => BAQPMaterial(),
                 "OSB" => OSBMaterial()]

function setup(N, mfak, α)
    Δx = LSTAB / N
    pc = PointCloud(LSTAB, Δx)
    params = MaterialParameters(; horizon=mfak * Δx, rho=RHO, E, alpha=α)
    # Last über eine Randschicht der Dicke δ; die Welle startet an deren Innenkante x0
    schicht = findall(p -> p ≤ params.δ, pc.position)
    bcs = [VelocityBC(t -> puls_v(t, VMAX, TPULS), schicht)]
    ref(x, t) = stab_analytisch(s -> puls_u(s, VMAX, TPULS), x, t; c=C0,
                                x0=pc.position[last(schicht)], l=LSTAB)
    return pc, params, bcs, ref
end

"rechnet die Modelle [label => mat, ...] mit gemeinsamem Δt; Rückgabe: runs, ref"
function rechne(modelle; N=1000, mfak=3.015, α=E, n_out=200)
    pc, params, bcs, ref = setup(N, mfak, α)
    system = BBVV1D.BondSystem(pc, BBMaterial(), params)
    Δt = BBVV1D.calc_stable_timestep(system, params)
    n_timesteps = ceil(Int, TEND / Δt)
    export_freq = max(1, n_timesteps ÷ n_out)
    runs = map(modelle) do (label, mat)
        hist = redirect_stdout(devnull) do
            simulation(pc, params, mat, bcs; n_timesteps, Δt, export_freq, vtk=false,
                       history=true)
        end
        (; label, hist, volume=pc.volume, rho=RHO)
    end
    return collect(runs), ref
end

maxfehler(run, ref) = maximum(fehler(run, ref))

function studie(modelle, werte; kwargs_von)
    fehler_tab = Dict(label => Float64[] for (label, _) in modelle)
    for w in werte
        rs, rf = rechne(modelle; kwargs_von(w)...)
        for r in rs
            push!(fehler_tab[r.label], maxfehler(r, rf))
        end
    end
    return fehler_tab
end

function drucke(titel, modelle, tab)
    println(titel)
    for (label, _) in modelle
        println("  ", rpad(label, 16), join((@sprintf("%.4f", e) for e in tab[label]), "  "))
    end
end

##-- Hauptlauf -------------------------------------------------------------------
printstyled(@sprintf("PMMA: c = %.1f m/s, Pulsdauer %.3e s, Laufzeit %.3e s\n", C0, TPULS,
                     TEND); color=:blue, bold=true)
runs, ref = rechne(MODELLE; n_out=1000)

speichern(plot_snapshots(runs, [0.3, 0.7, 1.1, 1.5, 1.9] .* TNORM; ref, tnorm=TNORM,
                         tlabel="t c/l"), joinpath(OUT, "02_momentaufnahmen"))
speichern(plot_xt(runs; ref, tnorm=TNORM, tlabel=TLABEL), joinpath(OUT, "03_xt"))
speichern(plot_probe(runs, [0.5LSTAB, LSTAB]; ref, tnorm=TNORM, tlabel=TLABEL),
          joinpath(OUT, "04_messpunkte"))
speichern(plot_error(runs, ref; tnorm=TNORM, tlabel=TLABEL), joinpath(OUT, "05_fehler"))
speichern(plot_energy(runs; tnorm=TNORM, tlabel=TLABEL), joinpath(OUT, "06_energie"))

println("max. relativer Fehler (N = 1000, δ = 3.015Δx):")
for r in runs
    @printf("  %s  %.4f\n", r.label, maxfehler(r, ref))
end

##-- Dispersion (analytisch) -----------------------------------------------------
Δx0 = LSTAB / 1000
kΔx = range(1e-3, π; length=400)
k = kΔx ./ Δx0
disp(m; kw...) = dispersion.(m, k; Δx=Δx0, δ=3.015Δx0, E, kw...)
speichern(plot_dispersion([("BB", kΔx, disp(:BB)), ("CC", kΔx, disp(:CCF)),
                           ("CC |ξ|", kΔx, disp(:CC)),
                           ("BAQP", kΔx, disp(:BAQP; kernel=const_one_kernel)),
                           ("OSB", kΔx, disp(:OSB))];
                          title=L"\delta = 3.015\,\Delta x,\ \mathrm{OSB\ mit}\ \omega = \delta/L"),
          joinpath(OUT, "01_dispersion"))
const KERNEL = ["konstant" => (const_one_kernel, :dash), "linear" => (linear_kernel, :solid),
                "kubischer B-Spline" => (cubic_b_spline_kernel, :dot)]
speichern(plot_dispersion([("OSB, ω " * name, kΔx, disp(:OSB; kernel=kern), FARBEN["OSB"], ls)
                           for (name, (kern, ls)) in KERNEL];
                          title=L"\mathrm{OSB},\ \delta = 3.015\,\Delta x,\ \alpha = E"),
          joinpath(OUT, "01b_dispersion_kernel"))

const ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
const ALPHA_FARBEN = ["#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal, blau
speichern(plot_dispersion([(@sprintf("α/E = %g", a), kΔx, disp(:OSB; α=a * E), c)
                           for (a, c) in zip(ALPHAS, ALPHA_FARBEN)];
                          title=L"\mathrm{OSB},\ \omega = \delta/L,\ \delta = 3.015\,\Delta x"),
          joinpath(OUT, "09a_dispersion_alpha"))

##-- Konvergenz in Δx (δ/Δx fest) ------------------------------------------------
Ns = [125, 250, 500, 1000, 2000]
dx_ticks = (LSTAB ./ Ns, ["l/$N" for N in Ns])
fehler_N = studie(MODELLE, Ns; kwargs_von=N -> (; N))
speichern(plot_study([(label, LSTAB ./ Ns, fehler_N[label]) for (label, _) in MODELLE];
                     xlabel=L"\Delta x", title=L"\delta = 3.015\,\Delta x", xticks=dx_ticks),
          joinpath(OUT, "07_konvergenz_dx"))

##-- Einfluss δ/Δx (Δx fest) ------------------------------------------------------
mfaks = [1.015, 2.015, 3.015, 4.015, 6.015]
fehler_m = studie(MODELLE, mfaks; kwargs_von=mf -> (; mfak=mf))
speichern(plot_study([(label, mfaks, fehler_m[label]) for (label, _) in MODELLE];
                     xlabel=L"\delta/\Delta x", xscale=identity, xticks=mfaks,
                     title=L"\Delta x = l/1000"),
          joinpath(OUT, "08_einfluss_horizont"))

##-- OSB: Einfluss α ---------------------------------------------------------------
fehler_α = studie(["OSB" => OSBMaterial()], ALPHAS; kwargs_von=a -> (; α=a * E))
speichern(plot_study([("OSB", ALPHAS, fehler_α["OSB"])]; xlabel=L"\alpha/E",
                     xticks=(ALPHAS, string.(ALPHAS)),
                     title=L"\mathrm{OSB},\ \Delta x = l/1000,\ \delta = 3.015\,\Delta x"),
          joinpath(OUT, "09b_einfluss_alpha"))

##-- Korrekturen BB / CC und Kernel OSB ----------------------------------------------
const KORREKTUREN = ["BB ohne Korrektur" => BBMaterial(),
                     "BB Energiekorrektur" => BBMaterial{EnergySurfaceCorrection}(),
                     "CC |ξ| Kontinuum" => CCMaterial(),
                     "CC |ξ| diskret" => CCMaterial(; calibration=:discrete)]
fehler_korr = studie(KORREKTUREN, Ns; kwargs_von=N -> (; N))
stil_korr = Dict("BB ohne Korrektur" => (FARBEN["BB"], :dash),
                 "BB Energiekorrektur" => (FARBEN["BB"], :solid),
                 "CC |ξ| Kontinuum" => (FARBEN["CC |ξ|"], :dash),
                 "CC |ξ| diskret" => (FARBEN["CC |ξ|"], :solid))
speichern(plot_study([(label, LSTAB ./ Ns, fehler_korr[label], stil_korr[label]...)
                      for (label, _) in KORREKTUREN];
                     xlabel=L"\Delta x", title=L"\delta = 3.015\,\Delta x", xticks=dx_ticks,
                     legendpos=:rb),
          joinpath(OUT, "10a_korrekturen"))

const KERNEL_MATS = ["OSB, ω " * name => OSBMaterial(; kernel=kern) for (name, (kern, _)) in KERNEL]
fehler_kern = studie(KERNEL_MATS, Ns; kwargs_von=N -> (; N))
speichern(plot_study([(label, LSTAB ./ Ns, fehler_kern[label], FARBEN["OSB"], ls)
                      for ((label, _), (_, (_, ls))) in zip(KERNEL_MATS, KERNEL)];
                     xlabel=L"\Delta x", title=L"\mathrm{OSB},\ \delta = 3.015\,\Delta x",
                     xticks=dx_ticks),
          joinpath(OUT, "10b_kernel"))

##-- Abgleich mit Peridynamics.jl (3D, dünner Stab 4×4 Punkte, ν = 0.25) ---------------
N3D = 500
runs_1d, ref3 = rechne(["BB" => BBMaterial{EnergySurfaceCorrection}(),
                        "OSB" => OSBMaterial()]; N=N3D, n_out=400)
pfad_3d = joinpath(OUT, "3d")
kw3d = (; l=LSTAB, N=N3D, E, rho=RHO, vmax=VMAX, T=TPULS, tend=TEND)
runs_3d = [stab_3d(Peridynamics.BBMaterial{Peridynamics.EnergySurfaceCorrection}(), "BB-3D";
                   kw3d..., path=joinpath(pfad_3d, "bb")),
           stab_3d(Peridynamics.OSBMaterial{Peridynamics.EnergySurfaceCorrection}(), "OSB-3D";
                   kw3d..., path=joinpath(pfad_3d, "osb"))]
runs_abgl = [runs_1d; runs_3d]
speichern(plot_snapshots(runs_abgl, [0.3, 0.7, 1.1, 1.5, 1.9] .* TNORM; ref=ref3,
                         tnorm=TNORM, tlabel="t c/l"),
          joinpath(OUT, "11a_abgleich_3d_momentaufnahmen"))
speichern(plot_probe(runs_abgl, [0.5LSTAB, LSTAB]; ref=ref3, tnorm=TNORM, tlabel=TLABEL),
          joinpath(OUT, "11b_abgleich_3d_messpunkte"))
speichern(plot_error(runs_abgl, ref3; tnorm=TNORM, tlabel=TLABEL),
          joinpath(OUT, "11c_abgleich_3d_fehler"))

##-- Zusammenfassung ---------------------------------------------------------------
println()
drucke("Konvergenz Δx, N = $(Ns) (max. Fehler):", MODELLE, fehler_N)
drucke("Einfluss δ/Δx = $(mfaks) (max. Fehler):", MODELLE, fehler_m)
println("OSB, α/E = ", ALPHAS, ":  ", join((@sprintf("%.4f", e) for e in fehler_α["OSB"]), "  "))
drucke("Korrekturen, N = $(Ns):", KORREKTUREN, fehler_korr)
drucke("Kernel OSB, N = $(Ns):", KERNEL_MATS, fehler_kern)
println("Abgleich 3D (N = $N3D, max. Fehler gegen d'Alembert):")
for r in runs_abgl
    @printf("  %-8s %.4f\n", r.label, maxfehler(r, ref3))
end
println("\nDiagramme in ", abspath(OUT))
