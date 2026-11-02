# Spaghetti (Weinberg: Breaking Spaghetti, Tab. 1): peridynamische Rechnung für das Paper.
# (a) statische Verformung des exzentrisch gedrückten Stabes bis zum Bruch gegen die Elastica (Tab. A2),
# (b) dynamische Entlastung des Bruchstücks: Anfangslage ist die rechte Hälfte der PD-Elastica im
#     Bruchzustand ab dem Scheitel, das ferne Ende gelenkig; alle Lasten durch die Bruchfläche werden
#     über die Risszeit T_F linear weggenommen. Krümmung über η = x/√(νt) gegen die Fresnel-Lösung.
#
#   julia --project=scripts/vergleich scripts/vergleich/spaghetti_pd.jl [ausgabepfad]

using BBVV1D, Printf, LinearAlgebra
include("plotting.jl")
include("balken_analytisch.jl")   # integralsinus

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "elastica")

# Tab. 1, SI-Einheiten
const L = 0.254
const D = 1.66e-3
const E = 3.375e9
const RHO = 1500.0
const SIGF = 30e6
const EI = E * π * D^4 / 64
const EA = E * π * D^2 / 4
const AQ = π * D^2 / 4
const WQ = π * D^3 / 32
const RHOA = RHO * AQ
const NU = sqrt(EI / RHOA)           # Biegekonstante ν = √(EI/ρA)
const FCR = π^2 * EI / L^2
const EXZ = 0.01 * L / 2
const NS = parse(Int, get(ENV, "SPAGHETTI_N", "1600"))
const MFAK = 3.015
const TF = 5e-6                      # Risszeit
const MODELL = Symbol(get(ENV, "SPAGHETTI_MODELL", "BP"))   # fürs Paper: BP

# Tab. A2: F/Fcr, w_max [mm]
const REF = [(0.25, 1.270), (0.50, 1.596), (0.80, 3.894), (0.90, 7.808), (0.95, 14.66), (1.00, 33.77),
             (1.10, 66.99)]

reale(sys, Y) = Y[:, (sys.nmax + 1):(sys.nmax + 1 + sys.N)]
wmax(sys, Y) = maximum(reale(sys, Y)[2, :])
randspannung(F, w) = -F / AQ + F * w / WQ

#-- (a) statisch ---------------------------------------------------------------------------
printstyled("statisch: $MODELL, N = $NS\n"; bold=true)
syss = ElasticaSystem(L, NS, MFAK * L / NS, EI, EA; rechts_frei=true, modell=MODELL)
fak = vcat(0.25:0.05:0.9, 0.92:0.02:1.1)
pfad = exzentrisch_pfad(syss, fak .* FCR; e=EXZ)
g(r) = randspannung(r.F, wmax(syss, r.Y)) - SIGF
r1, r2 = pfad[end-2], pfad[end-1]          # 1.06 und 1.08 F_cr, Bruch dazwischen bis 1.1
for _ in 1:10
    F3 = r2.F - g(r2) * (r2.F - r1.F) / (g(r2) - g(r1))
    r3 = only(exzentrisch_pfad(syss, [F3]; e=EXZ, q0=r2.q))
    global r1, r2 = r2, r3
    abs(g(r2)) < 1e-7 * SIGF && break
end
bruch = r2
Yb = reale(syss, bruch.Y)
κb = elastica_kruemmung(syss, bruch.Y)
@printf("Bruch: F/Fcr = %.4f, w_max = %.2f mm, κ_max = %.2f 1/m\n", bruch.F / FCR, 1e3 * wmax(syss, bruch.Y),
        maximum(abs, κb))

#-- (b) Bruchstück -----------------------------------------------------------------------
kc = argmax(Yb[2, :]) - 1                     # Scheitelknoten (0-basiert)
NF = NS - kc
Δs = syss.Δs
sysf = ElasticaSystem(NF * Δs, NF, MFAK * Δs, EI, EA; rechts_frei=true, modell=MODELL)
# Knoten i des Bruchstücks = Knoten NS - i des Stabes; Knoten 0 (fernes Ende) im Ursprung, gelenkig
q0 = vec(hcat([Yb[:, NS - i + 1] .- Yb[:, NS + 1] for i in 1:NF]...))
@printf("Bruchstück: Länge %.1f mm, %d Knoten\n", 1e3 * NF * Δs, NF)

# ∂W/∂q des Bruchstücks; der Geisterknoten -n (Spalte nmax + 1 - n) ist -y_n
function gradient(q)
    Y = exzentrisch_lagen(sysf, q)
    G = similar(Y)
    BBVV1D.elastica_energie!(G, sysf, Y)
    Gr = G[:, (sysf.nmax + 2):(sysf.nmax + 1 + NF)]
    for n in 1:sysf.nmax
        Gr[:, n] .-= G[:, sysf.nmax + 1 - n]
    end
    return vec(Gr)
end

fhalt = gradient(q0)                  # Haltekräfte: Gleichgewicht bei t = 0
M = repeat(RHOA .* sysf.volume[2:NF+1]; inner=2)
K = BBVV1D.tangente_fd(gradient, q0, NF, 2sysf.nmax, 1e-7 * sysf.L)
s = 1 ./ sqrt.(M)
v = randn(length(q0)); λ = 0.0
for _ in 1:80
    u = s .* (K * (s .* v)); global λ = norm(u) / norm(v); global v = u ./ norm(u)
end
Δt = 0.8 * 2 / sqrt(λ)
const ZEITEN = [25e-6, 50e-6, 100e-6]       # vor der Rückkehr der kurzen Wellen vom fernen Ende
tend = last(ZEITEN)
nsteps = ceil(Int, tend / Δt)
Δt = tend / nsteps
@printf("ω_max = %.3e 1/s, Δt = %.3e s, %d Schritte\n", sqrt(λ), Δt, nsteps)
ausgabe = Set(round(Int, t / Δt) for t in ZEITEN)

κ_init = elastica_kruemmung(sysf, exzentrisch_lagen(sysf, q0))
x = (NF .- (0:NF)) .* Δs                      # Abstand von der Bruchfläche
κ0 = maximum(abs, κb)                         # Scheitelkrümmung im Bruchzustand
profile = Dict{Float64,Vector{Float64}}()
q = copy(q0); vel = zeros(length(q0))
a = (fhalt .- gradient(q)) ./ M
E0 = 0.0
for step in 1:nsteps
    t = step * Δt
    @. vel += 0.5Δt * a
    @. q += Δt * vel
    f = fhalt .* max(0.0, 1 - t / TF)
    a .= (f .- gradient(q)) ./ M
    @. vel += 0.5Δt * a
    if step in ausgabe
        profile[t] = elastica_kruemmung(sysf, exzentrisch_lagen(sysf, q))
    end
    if step == ceil(Int, TF / Δt)
        global E0 = elastica_energie(sysf, exzentrisch_lagen(sysf, q)) + 0.5sum(M .* vel .^ 2)
    end
end
Eend = elastica_energie(sysf, exzentrisch_lagen(sysf, q)) + 0.5sum(M .* vel .^ 2)
@printf("Energie nach dem Loslassen: relative Änderung %.1e\n", (Eend - E0) / E0)

fresnel(η) = sqrt(2 / π) * simpson(s -> sin(s^2 / 4), 0, η)
function simpson(f, a, b; n=400)
    h = (b - a) / 2n
    return h / 3 * (f(a) + f(b) + 4sum(f(a + (2k - 1)h) for k in 1:n) + 2sum(f(a + 2k * h) for k in 1:n-1))
end
for t in sort(collect(keys(profile)))
    u = abs.(profile[t]) ./ κ0
    η = x ./ sqrt(NU * (t - TF / 2))   # Loslassen im Mittel bei T_F/2
    sel = (η .>= 1) .& (η .<= 5)
    k = findall(sel)[argmax(u[sel])]
    @printf("t = %6.0f µs: max κ/κ0 = %.3f bei η = %.2f, x = %.1f mm, κ(x,0)/κ0 = %.3f, κ/κ(x,0) = %.3f (Fresnel 1.428 bei 3.545)\n",
            1e6t, u[k], η[k], 1e3x[k], abs(κ_init[k]) / κ0, u[k] * κ0 / abs(κ_init[k]))
end

#-- Bild -----------------------------------------------------------------------------------
fig = with_theme(THEMA) do
    f = Figure(size=(900, 340))
    ax = Axis(f[1, 1]; xlabel=L"F/F_\mathrm{cr}", ylabel=L"w_\mathrm{max}\ \mathrm{[mm]}",
              title="(a) static loading")
    lines!(ax, [r.F / FCR for r in pfad], [1e3 * wmax(syss, r.Y) for r in pfad]; color=FARBEN["BP"],
           label="peridynamics")
    scatter!(ax, first.(REF), last.(REF); color=REF_FARBE, marker=:circle, markersize=9,
             label="elastica, Table A2")
    scatter!(ax, [1.0891], [64.49]; color=REF_FARBE, marker=:star5, markersize=14, label="fracture")
    axislegend(ax; position=:lt)
    ax2 = Axis(f[1, 2]; xlabel=L"\eta = x/\sqrt{\nu t}", ylabel=L"\kappa/\kappa_0",
               title="(b) release of the fragment")
    ηs = range(0, 7; length=400)
    lines!(ax2, ηs, fresnel.(ηs); color=REF_FARBE, linewidth=2.5, label="Fresnel (30)")
    farben = (FARBEN["BB"], FARBEN["CC"], FARBEN["OSB"], FARBEN["CC |ξ|"])
    for (i, t) in enumerate(sort(collect(keys(profile))))
        η = x ./ sqrt(NU * (t - TF / 2))   # Loslassen im Mittel bei T_F/2
        sel = η .<= 7
        lines!(ax2, η[sel], (abs.(profile[t]) ./ κ0)[sel]; color=farben[i], linewidth=1.5, linestyle=:dash,
               label=Makie.LaTeXString(@sprintf("\$t = %.0f\\,\\mu\\mathrm{s}\$", 1e6t)))
    end
    hlines!(ax2, [1.428]; color=(REF_FARBE, 0.4), linestyle=:dot)
    limits!(ax2, 0, 7, -0.1, 1.6)
    axislegend(ax2; position=:rb)
    f
end
speichern(fig, joinpath(OUT, "spaghetti_pd_$(lowercase(String(MODELL)))"))
println("Bild in ", abspath(OUT))
