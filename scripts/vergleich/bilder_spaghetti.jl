# Partikelbilder der Spaghetti (Weinberg: Breaking Spaghetti) mit der BP-Elastica (GLMakie):
# links der exzentrisch gedrückte Stab der Länge 2ℓ im Bruchzustand (grob, damit die Partikel
# sichtbar sind), rechts die ersten Zentimeter des Bruchstücks an der Bruchfläche kurz nach dem
# Loslassen (fein gerechnet): rechte Hälfte ab dem Scheitel, fernes Ende gelenkig, Entlastung über
# T_F. Farbe κ/κ₀ mit der Scheitelkrümmung κ₀ im Bruchzustand.
#
#   julia --project=scripts/vergleich scripts/vergleich/bilder_spaghetti.jl [ausgabepfad] [zeiten in µs]

using BBVV1D, GLMakie, LinearAlgebra, Printf
include("kugeln.jl")
GLMakie.activate!(visible=false)

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "bilder", "elastica")
ZEITEN_US = length(ARGS) ≥ 2 ? parse.(Float64, split(ARGS[2], ",")) : [5.0, 30.0]
mkpath(OUT)

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
const FCR = π^2 * EI / L^2
const EXZ = 0.01 * L / 2
const N_GROB = 100            # Bild des ganzen Stabes
const N_FEIN = 400            # Rechnung des Bruchstücks
const TF = 5e-6
const AUSSCHNITT = 0.05       # gezeigte Länge ab der Bruchfläche [m]

reale(sys, Y) = Y[:, (sys.nmax + 1):(sys.nmax + 1 + sys.N)]
wmax(sys, Y) = maximum(reale(sys, Y)[2, :])

"statische Lösung im Bruchzustand (Randspannung σ_F am Scheitel)"
function bruchzustand(N)
    sys = ElasticaSystem(L, N, 3.015L / N, EI, EA; rechts_frei=true, modell=:BP)
    pfad = exzentrisch_pfad(sys, vcat(0.25:0.05:0.9, 0.92:0.02:1.1) .* FCR; e=EXZ)
    g(r) = -r.F / AQ + r.F * wmax(sys, r.Y) / WQ - SIGF
    r1, r2 = pfad[end-2], pfad[end-1]
    for _ in 1:10
        F3 = r2.F - g(r2) * (r2.F - r1.F) / (g(r2) - g(r1))
        r1, r2 = r2, only(exzentrisch_pfad(sys, [F3]; e=EXZ, q0=r2.q))
        abs(g(r2)) < 1e-7 * SIGF && break
    end
    return sys, r2
end

sysg, bruchg = bruchzustand(N_GROB)
Yg = reale(sysg, bruchg.Y)
κg = elastica_kruemmung(sysg, bruchg.Y)
syss, bruch = bruchzustand(N_FEIN)
Yb = reale(syss, bruch.Y)
κ0 = maximum(abs, elastica_kruemmung(syss, bruch.Y))
@printf("Bruch bei F = %.4f F_cr (grob %.4f), κ₀ = %.2f 1/m\n", bruch.F / FCR, bruchg.F / FCR, κ0)

# Bruchstück: Knoten i = Knoten N - i des Stabes, Knoten 0 (fernes Ende) gelenkig
kc = argmax(Yb[2, :]) - 1
NF = N_FEIN - kc
Δs = syss.Δs
sysf = ElasticaSystem(NF * Δs, NF, 3.015Δs, EI, EA; rechts_frei=true, modell=:BP)
ursprung = Yb[:, N_FEIN + 1]
q0 = vec(hcat([Yb[:, N_FEIN - i + 1] .- ursprung for i in 1:NF]...))
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
fhalt = gradient(q0)
M = repeat(RHOA .* sysf.volume[2:NF+1]; inner=2)
K = BBVV1D.tangente_fd(gradient, q0, NF, 2sysf.nmax, 1e-7 * sysf.L)
s = 1 ./ sqrt.(M); v = randn(length(q0)); λ = 0.0
for _ in 1:80
    u = s .* (K * (s .* v)); global λ = norm(u) / norm(v); global v = u ./ norm(u)
end
tend = 1e-6 * maximum(ZEITEN_US)
Δt = min(0.8 * 2 / sqrt(λ), TF / 20)
nsteps = ceil(Int, tend / Δt); Δt = tend / nsteps
ausgabe = Dict(round(Int, 1e-6t / Δt) => t for t in ZEITEN_US)
lagen = Dict{Float64,Matrix{Float64}}()
q = copy(q0); vel = zeros(length(q0))
a = (fhalt .- gradient(q)) ./ M
for step in 1:nsteps
    t = step * Δt
    @. vel += 0.5Δt * a
    @. q += Δt * vel
    a .= (fhalt .* max(0.0, 1 - t / TF) .- gradient(q)) ./ M
    @. vel += 0.5Δt * a
    haskey(ausgabe, step) && (lagen[ausgabe[step]] = exzentrisch_lagen(sysf, q))
end
@printf("Bruchstück: %d Partikel, %d Zeitschritte\n", NF, nsteps)

# Punkte im System des ganzen Stabes; Ausschnitt = Knoten mit Abstand ≤ AUSSCHNITT von der Bruchfläche
stab_pkt = [Point3f(Yg[1, k], Yg[2, k], 0) for k in 1:N_GROB+1]
nah = [i for i in 0:NF if (NF - i) * Δs <= AUSSCHNITT]
frag_pkt(Y) = [Point3f(Y[1, sysf.nmax + 1 + i] + ursprung[1], Y[2, sysf.nmax + 1 + i] + ursprung[2], 0) for i in nah]
Y0 = exzentrisch_lagen(sysf, q0)
p0 = frag_pkt(Y0)
# Verschiebungen gegenüber dem Bruchzustand vergrößert, damit die Formänderung sichtbar wird
const VERGR = parse(Float64, get(ENV, "SPAGHETTI_VERGR", "1"))
frag_vergr(Y) = frag_pkt(Y0 .+ VERGR .* (Y .- Y0))
alle = vcat(p0, [frag_vergr(Y) for Y in values(lagen)]...)
xa, ya = getindex.(alle, 1), getindex.(alle, 2)
ra = 0.06AUSSCHNITT
gr_a = (minimum(xa) - ra, maximum(xa) + ra, minimum(ya) - 1.5ra, maximum(ya) + ra)
rg = 0.04L
gr_g = (minimum(Yg[1, :]) - rg, maximum(Yg[1, :]) + rg, minimum(Yg[2, :]) - rg, maximum(Yg[2, :]) + rg)

fig = with_theme(merge(Theme(fontsize=50, Axis=(titlefont=:regular,)), KUGEL_THEMA)) do
    zeiten = sort(collect(keys(lagen)))
    f = Figure(size=(2400, 900))
    ax = kugelachse(f[1:length(zeiten), 1]; title="at fracture")
    ms = kugeln!(ax, stab_pkt, abs.(κg) ./ κ0; radius=0.45sysg.Δs, colorrange=(0, 1.5))
    lines!(ax, [gr_a[1], gr_a[2], gr_a[2], gr_a[1], gr_a[1]], [gr_a[3], gr_a[3], gr_a[4], gr_a[4], gr_a[3]];
           color=:gray40, linewidth=2)
    grenzen!(ax, gr_g, 0)
    for (j, t) in enumerate(zeiten)
        Y = lagen[t]
        κ = abs.(elastica_kruemmung(sysf, Y))[nah .+ 1] ./ κ0
        axj = kugelachse(f[j, 2])
        kugeln!(axj, frag_vergr(Y), κ; radius=0.5Δs, colorrange=(0, 1.5))
        text!(axj, gr_a[1] + 0.12(gr_a[2] - gr_a[1]), gr_a[3]; text=@sprintf("t = %.0f μs", t),
              align=(:left, :bottom))
        grenzen!(axj, gr_a, 0)
    end
    Colorbar(f[1:length(zeiten), 3], ms; label=L"\kappa/\kappa_0", height=Relative(0.6))
    colsize!(f.layout, 1, Relative(0.55))
    f
end
for t in sort(collect(keys(lagen)))
    d = reale(sysf, lagen[t] .- Y0)
    @printf("t = %.0f µs: größte Verschiebung %.3f mm\n", t, 1e3 * maximum(sqrt.(sum(d .^ 2; dims=1))))
end
name = @sprintf("04_spaghetti_x%.0f", VERGR)
save(joinpath(OUT, name * ".png"), fig; px_per_unit=1)
println("Bild ", joinpath(OUT, name * ".png"))
