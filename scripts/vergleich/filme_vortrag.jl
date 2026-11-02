# Filme für den Vortrag (englische Titel): Loslassen eines Kreisbogens wie in bilder_elastica.jl und
# Loslassen des Spaghetti-Bruchstücks als Kragarm: rechte Hälfte der BP-Elastica im Bruchzustand ab
# dem Scheitel, das ferne Ende eingespannt, die Haltekräfte über T_F linear weggenommen.
#
#   julia --project=scripts/vergleich scripts/vergleich/filme_vortrag.jl [ausgabepfad]

using BBVV1D, GLMakie, LinearAlgebra, Printf
include("kugeln.jl")
GLMakie.activate!(visible=false)

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "bilder", "elastica")
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
const CB = sqrt(EI / RHOA)
const EXZ = 0.01 * L / 2
const TF = 5e-6

"Zeitschritt aus der größten Eigenfrequenz (Potenzmethode auf M⁻¹K mit Differenzen)"
function zeitschritt(grad, q0, M)
    h = 1e-9 * L
    v = randn(length(q0))
    λ = 0.0
    for _ in 1:60
        u = (grad(q0 .+ h .* v) .- grad(q0 .- h .* v)) ./ (2h) ./ M
        λ = norm(u) / norm(v)
        v = u ./ norm(u)
    end
    return 0.5 * 2 / sqrt(λ)
end

##-- Loslassen eines Kreisbogens --------------------------------------------------------------
println("Kreisbogen")
sysk = ElasticaSystem(L, 200, 3.015L / 200, EI, EA; rechts_frei=true)
κ0 = 4π / L
qk = kragarm_bogen(sysk, κ0)
Mk = repeat(RHOA .* sysk.volume[2:end]; inner=2)
tendk = (L / 8)^2 / CB
h = kragarm_dynamik(sysk, RHOA; q0=qk, tend=tendk, Δt=zeitschritt(q -> kragarm_gradient(sysk, q), qk, Mk),
                    n_out=300, endmoment=t -> EI * κ0 * max(0.0, 1 - t / 25e-6))
elastica_film(joinpath(OUT, "05_release_arc.mp4"), sysk, [kragarm_lagen(sysk, q) for q in h.q],
              [k ./ κ0 for k in h.κ], [@sprintf("release of a circular arc, κ₀L = 4π:  t = %.2f ms", 1e3t) for t in h.time];
              radius=0.45sysk.Δs, colorrange=(0, 1.5), label=L"\kappa/\kappa_0", fps=30)
println("  05_release_arc.mp4")

##-- Bruchstück als Kragarm -------------------------------------------------------------------
println("Kragarm")
const NS = 200
reale(sys, Y) = Y[:, (sys.nmax + 1):(sys.nmax + 1 + sys.N)]
wmax(sys, Y) = maximum(reale(sys, Y)[2, :])
sys = ElasticaSystem(L, NS, 3.015L / NS, EI, EA; rechts_frei=true, modell=:BP)
pfad = exzentrisch_pfad(sys, vcat(0.25:0.05:0.9, 0.92:0.02:1.1) .* FCR; e=EXZ)
g(r) = -r.F / AQ + r.F * wmax(sys, r.Y) / WQ - SIGF
r1, r2 = pfad[end-2], pfad[end-1]
for _ in 1:10
    F3 = r2.F - g(r2) * (r2.F - r1.F) / (g(r2) - g(r1))
    global r1, r2 = r2, only(exzentrisch_pfad(sys, [F3]; e=EXZ, q0=r2.q))
    abs(g(r2)) < 1e-7 * SIGF && break
end
Yb = reale(sys, r2.Y)
κb = maximum(abs, elastica_kruemmung(sys, r2.Y))
kc = argmax(Yb[2, :]) - 1
NF = NS - kc
# Knoten i des Bruchstücks = Knoten NS - i des Stabes; Einspannung im Ursprung, Tangente in x-Richtung
P = [Yb[:, NS - i + 1] .- Yb[:, NS + 1] for i in 0:NF]
θ = atan(P[2][2], P[2][1])
R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
P = [R * p for p in P]
sysf = ElasticaSystem(NF * sys.Δs, NF, 3.015sys.Δs, EI, EA; rechts_frei=true, modell=:BP)
q0 = vec(hcat(P[2:end]...))
fhalt = kragarm_gradient(sysf, q0)
M = repeat(RHOA .* sysf.volume[2:end]; inner=2)
Δt = zeitschritt(q -> kragarm_gradient(sysf, q), q0, M)
tend = 0.25 * (NF * sysf.Δs)^2 / CB
nsteps = ceil(Int, tend / Δt); Δt = tend / nsteps
nbild = 360; freq = max(1, nsteps ÷ nbild)
q = copy(q0); v = zeros(length(q))
a = (fhalt .- kragarm_gradient(sysf, q)) ./ M
Ys = [kragarm_lagen(sysf, q)]; zeiten = [0.0]
for step in 1:nsteps
    t = step * Δt
    @. v += 0.5Δt * a
    @. q += Δt * v
    a .= (fhalt .* max(0.0, 1 - t / TF) .- kragarm_gradient(sysf, q)) ./ M
    @. v += 0.5Δt * a
    if step % freq == 0
        push!(Ys, kragarm_lagen(sysf, q)); push!(zeiten, t)
    end
end
@printf("Bruchstück: %d Partikel, %d Schritte, Ende %.2f ms\n", NF, nsteps, 1e3tend)
κs = [abs.(elastica_kruemmung(sysf, Y)) ./ κb for Y in Ys]
elastica_film(joinpath(OUT, "06_release_cantilever.mp4"), sysf, Ys, κs,
              [@sprintf("released fragment, far end clamped:  t = %.2f ms", 1e3t) for t in zeiten];
              radius=0.45sysf.Δs, colorrange=(0, 1.5), label=L"\kappa/\kappa_0", fps=30)
println("  06_release_cantilever.mp4")
