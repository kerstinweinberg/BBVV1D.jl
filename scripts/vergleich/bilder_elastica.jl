# Partikelbilder und Filme der peridynamischen Elastica (GLMakie): Nachknicken,
# Durchschlagen (asymmetrisch und symmetrisch) und Loslassen eines gekrümmten Kragarms.
# Gröbere Teilung als in den Vergleichsrechnungen, damit die Partikel sichtbar bleiben.
#
#   julia --project=scripts/vergleich scripts/vergleich/bilder_elastica.jl [ausgabepfad]

using BBVV1D, GLMakie, LinearAlgebra, Printf
include("kugeln.jl")
GLMakie.activate!(visible=false)

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "bilder", "elastica")
mkpath(OUT)
bild(fig, name) = (save(joinpath(OUT, name * ".png"), fig; px_per_unit=2); println("  ", name, ".png"))

# Knickstab wie in knickstab.jl, SI-Einheiten
const E = 3.375e9
const RHO = 1500.0
const D = 1.66e-3
const L = 0.254
const EI = E * π * D^4 / 64
const EA = E * π * D^2 / 4
const RHOA = RHO * π * D^2 / 4
const FCR = π^2 * EI / L^2
const CB = sqrt(EI / RHOA)

# Zeitschritt aus der größten Eigenfrequenz (Potenzmethode auf M⁻¹K mit Differenzen)
function zeitschritt(grad, q0, M; L=L)
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

##-- 1. Nachknicken -------------------------------------------------------------------------
println("Nachknicken")
sys = ElasticaSystem(L, 60, 3.015L / 60, EI, EA)
Δs = range(0.002, 1.45; length=150) .* L
pfad = elastica_pfad(sys, Δs)
# Krümmung ist gegen den Uhrzeigersinn positiv, der nach oben geknickte Stab hat κ < 0
κL = [abs.(elastica_kruemmung(sys, r.Y)) .* L for r in pfad]
κmax = maximum(maximum.(κL))
rad = 0.45sys.Δs
titel(r) = rich(@sprintf("Δ/L = %.2f,  F/F", r.Δ / L), subscript("cr"), @sprintf(" = %.2f", r.F / FCR))
fig = with_theme(KUGEL_THEMA) do
    f = Figure(size=(1200, 400))
    ms = nothing
    for (j, Δr) in enumerate((0.25, 0.75, 1.0, 1.4))
        k = argmin(abs.(Δs ./ L .- Δr))
        ax = kugelachse(f[1, j]; title=titel(pfad[k]))
        ms = kugeln!(ax, elastica_punkte(sys, pfad[k].Y), κL[k]; radius=rad, colorrange=(0, κmax))
    end
    Colorbar(f[1, 5], ms; label=L"|\kappa| L", height=Relative(0.6))
    f
end
bild(fig, "01_nachknicken")
elastica_film(joinpath(OUT, "01_nachknicken.mp4"), sys, [r.Y for r in pfad], κL, titel.(pfad);
              radius=rad, colorrange=(0, κmax), label=L"|\kappa| L", fps=25)
println("  01_nachknicken.mp4")

##-- 2. Durchschlagen -----------------------------------------------------------------------
println("Durchschlagen")
const DELTA = 0.05L
sysd = ElasticaSystem(L, 60, 3.015L / 60, EI, EA)
q0 = elastica_pfad(sysd, range(0.001L, DELTA; length=10))[end].q
# statische Durchschlagkraft (Verzweigung) wie in vergleich_durchschlag.jl
function λmin(Q)
    q, _ = elastica_gleichgewicht(sysd, q0, DELTA; Q=-Q * FCR)
    minimum(eigvals(Symmetric(Matrix(BBVV1D.elastica_tangente(sysd, q, DELTA; Q=-Q * FCR)))))
end
a, b = 0.0, 5.0
for _ in 1:30
    global a, b
    m = (a + b) / 2
    λmin(m) > 0 ? (a = m) : (b = m)
end
Qs = (a + b) / 2
Md = elastica_massen(sysd, RHOA)
Δtd = zeitschritt(q -> elastica_residuum(sysd, q, DELTA)[1], q0, Md)
T1 = 0.0202
for (fak, tend, name) in ((1.5, 3T1, "02a_durchschlag_symmetrisch"),
                          (1.2, 8T1, "02b_durchschlag_asymmetrisch"))
    qa = copy(q0)
    for i in 1:sysd.N-1
        qa[2i] += 1e-5L * sin(2π * i / sysd.N)
    end
    h = elastica_dynamik(sysd, DELTA, RHOA; q0=qa, Q=t -> -fak * Qs * FCR, tend, Δt=Δtd, n_out=300)
    Ys = [elastica_lagen(sysd, q, DELTA) for q in h.q]
    κ = [elastica_kruemmung(sysd, Y) .* L for Y in Ys]
    km = maximum(maximum(abs, k) for k in κ)
    radd = 0.45sysd.Δs
    zeiten = range(0, tend; length=6)
    fig = with_theme(KUGEL_THEMA) do
        f = Figure(size=(1400, 640))
        ms = nothing
        for (j, t) in enumerate(zeiten)
            k = argmin(abs.(h.time .- t))
            ax = kugelachse(f[(j - 1) ÷ 2 + 1, (j - 1) % 2 + 1]; title=@sprintf("t/T₁ = %.2f", h.time[k] / T1))
            ms = kugeln!(ax, elastica_punkte(sysd, Ys[k]), κ[k]; radius=radd, colorrange=(-km, km))
            g = grenzen(sysd, Ys)
            grenzen!(ax, g, radd)
        end
        Colorbar(f[1:3, 3], ms; label=L"\kappa L", height=Relative(0.6))
        Label(f[0, :], @sprintf("Sprunglast Q = %.1f Qₛ in der Mitte, Δ = 0.05 L", fak); fontsize=16)
        f
    end
    bild(fig, name)
    elastica_film(joinpath(OUT, name * ".mp4"), sysd, Ys, κ,
                  [@sprintf("Q = %.1f Qₛ:  t/T₁ = %.2f", fak, t / T1) for t in h.time];
                  radius=radd, colorrange=(-km, km), label=L"\kappa L", fps=30, size=(1000, 400))
    println("  ", name, ".mp4")
end

##-- 3. Loslassen eines gekrümmten Kragarms -------------------------------------------------
println("Loslassen")
sysk = ElasticaSystem(L, 200, 3.015L / 200, EI, EA; rechts_frei=true)
κ0 = 4π / L
qk = kragarm_bogen(sysk, κ0)
Mk = repeat(RHOA .* sysk.volume[2:end]; inner=2)
Δtk = zeitschritt(q -> kragarm_gradient(sysk, q), qk, Mk)
tendk = (L / 8)^2 / CB
h = kragarm_dynamik(sysk, RHOA; q0=qk, tend=tendk, Δt=Δtk, n_out=300,
                    endmoment=t -> EI * κ0 * max(0.0, 1 - t / 25e-6))
Ys = [kragarm_lagen(sysk, q) for q in h.q]
κr = [k ./ κ0 for k in h.κ]
radk = 0.45sysk.Δs
fig = with_theme(KUGEL_THEMA) do
    f = Figure(size=(1200, 380))
    ms = nothing
    g = grenzen(sysk, Ys)
    for (j, fr) in enumerate((0.0, 0.1, 0.3, 1.0))
        k = argmin(abs.(h.time .- fr * tendk))
        ax = kugelachse(f[1, j]; title=@sprintf("t = %.2f ms", 1e3h.time[k]))
        ms = kugeln!(ax, elastica_punkte(sysk, Ys[k]), κr[k]; radius=radk, colorrange=(0, 1.5))
        grenzen!(ax, g, radk)
    end
    Colorbar(f[1, 5], ms; label=L"\kappa/\kappa_0", height=Relative(0.6))
    f
end
bild(fig, "03_loslassen")
elastica_film(joinpath(OUT, "03_loslassen.mp4"), sysk, Ys, κr,
              [@sprintf("Loslassen, κ₀L = 4π:  t = %.3f ms", 1e3t) for t in h.time];
              radius=radk, colorrange=(0, 1.5), label=L"\kappa/\kappa_0", fps=30)
println("  03_loslassen.mp4")
println("\nBilder und Filme in ", abspath(OUT))
