# Dynamisches Durchschlagen des geknickten Stabes (peridynamische Elastica):
# Knickstab um Δ verkürzt und nach oben ausgeknickt, dann schlagartig eine Querkraft Q in der
# Mitte nach unten. Statische Durchschlagkraft Q_s aus dem Stabilitätsverlust (kleinster
# Eigenwert der Tangente), Zeitverläufe, Gestalt, Energiebilanz, Schwelle, Konvergenz in N.
#
#   julia --project=scripts/vergleich scripts/vergleich/vergleich_durchschlag.jl [ausgabepfad]

using BBVV1D, LinearAlgebra, Printf
include("plotting.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "durchschlag")

# Knickstab wie in knickstab.jl, SI-Einheiten
const E = 3.375e9           # Pa
const RHO = 1500.0          # kg/m³
const D = 1.66e-3           # m
const L = 0.254             # m
const EI = E * π * D^4 / 64
const EA = E * π * D^2 / 4
const RHOA = RHO * π * D^2 / 4
const FCR = π^2 * EI / L^2
const DELTA = 0.05L         # Verkürzung
const STOER = 1e-5L         # asymmetrische Anfangsstörung (sin 2πX/L)
const TEND = 0.3

function aufbau(N)
    sys = ElasticaSystem(L, N, 3.015L / N, EI, EA)
    q0 = elastica_pfad(sys, range(0.001L, DELTA; length=10))[end].q
    return sys, q0
end

"statische Durchschlagkraft: Q, bei dem die Tangente auf dem symmetrischen Ast singulär wird"
function durchschlagkraft(sys, q0)
    λmin(Q) = begin
        q, _ = elastica_gleichgewicht(sys, q0, DELTA; Q=-Q * FCR)
        minimum(eigvals(Symmetric(Matrix(BBVV1D.elastica_tangente(sys, q, DELTA; Q=-Q * FCR)))))
    end
    a, b = 0.0, 5.0
    for _ in 1:40
        m = (a + b) / 2
        λmin(m) > 0 ? (a = m) : (b = m)
    end
    return (a + b) / 2
end

function lauf(sys, q0, Qfak, Qs; tend=TEND)
    qa = copy(q0)
    for i in 1:sys.N-1
        qa[2i] += STOER * sin(2π * i / sys.N)
    end
    Δt = 0.5 * 2 / elastica_omega_max(sys, q0, DELTA, RHOA)
    h = elastica_dynamik(sys, DELTA, RHOA; q0=qa, Q=t -> -Qfak * Qs * FCR, tend, Δt, n_out=1500)
    k = findfirst(<(0), h.zmitte)
    return (; h, t_durch=k === nothing ? NaN : h.time[k], Δt)
end

N0 = 100
sys, q0 = aufbau(N0)
Qs = durchschlagkraft(sys, q0)
K0 = BBVV1D.elastica_tangente(sys, q0, DELTA)
M0 = elastica_massen(sys, RHOA)
ω1 = sqrt(minimum(eigvals(Symmetric(Matrix(Diagonal(1 ./ sqrt.(M0)) * K0 *
                                             Diagonal(1 ./ sqrt.(M0)))))))
T1 = 2π / ω1
printstyled(@sprintf("Knickstab Δ = %.2f L: z_mitte = %.4f L, ω₁ = %.1f rad/s, T₁ = %.4f s\n",
                     DELTA / L, q0[2(N0 ÷ 2)] / L, ω1, T1); color=:blue, bold=true)
@printf("statische Durchschlagkraft Q_s = %.4f F_cr = %.4e N\n", Qs, Qs * FCR)

"Grenzlast des symmetrischen Astes (ohne Störung bleibt die Lösung symmetrisch)"
function symmetrische_grenzlast(sys, q0; Qa=2.0, dQ=0.01)
    q = copy(q0)
    for Q in Qa:dQ:10.0
        qn, _ = redirect_stderr(devnull) do
            elastica_gleichgewicht(sys, q, DELTA; Q=-Q * FCR)
        end
        qn[2(sys.N ÷ 2)] < 0 && return Q - dQ / 2
        q = qn
    end
    return NaN
end
QL = symmetrische_grenzlast(sys, q0)
@printf("Grenzlast des symmetrischen Astes Q_L = %.3f F_cr = %.3f Qₛ\n", QL, QL / Qs)
println("dynamisch symmetrisch (ohne Störung), Durchschlag bis 0.12 s:")
for f in (1.20, 1.25, 1.30)
    qs = copy(q0)
    Δt = 0.5 * 2 / elastica_omega_max(sys, q0, DELTA, RHOA)
    h = elastica_dynamik(sys, DELTA, RHOA; q0=qs, Q=t -> -f * Qs * FCR, tend=0.12, Δt, n_out=600)
    k = findfirst(<(0), h.zmitte)
    @printf("  Q = %.2f Qₛ = %.2f Q_L: %s\n", f, f * Qs / QL, k === nothing ? "kein Durchschlag" :
                                                    @sprintf("t = %.4f s", h.time[k]))
end

##-- Konvergenz der statischen Durchschlagkraft --------------------------------------------
println("Q_s/F_cr über N:")
for N in (50, 100, 200)
    s, q = aufbau(N)
    @printf("  N = %3d: %.4f\n", N, durchschlagkraft(s, q))
end

##-- Zeitverläufe -----------------------------------------------------------------------------
fake = [1.0, 1.1, 1.2, 1.5, 2.0]
farben = ["#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]     # ordinal, blau
laeufe = [lauf(sys, q0, f, Qs) for f in fake]
fig = with_theme(THEMA) do
    f = Figure(size=(720, 340))
    ax = Axis(f[1, 1]; xlabel=L"t/T_1", ylabel=L"z_\mathrm{mitte}/L",
              title=@sprintf("Sprunglast Q in der Mitte, Δ = %.2f L, N = %d", DELTA / L, N0))
    for (fk, r, c) in zip(fake, laeufe, farben)
        lines!(ax, r.h.time ./ T1, r.h.zmitte ./ L; color=c, linewidth=1.5,
               label=@sprintf("Q = %.1f Qₛ", fk))
    end
    hlines!(ax, 0.0; color=(:black, 0.3), linewidth=1)
    Legend(f[1, 2], ax)
    f
end
speichern(fig, joinpath(OUT, "01_zeitverlauf"))
println("Durchschlag (erster Nulldurchgang der Mitte):")
for (fk, r) in zip(fake, laeufe)
    @printf("  Q = %.1f Q_s: t = %s\n", fk, isnan(r.t_durch) ? "kein Durchschlag bis 0.3 s" :
                                          @sprintf("%.4f s = %.2f T₁", r.t_durch, r.t_durch / T1))
end

##-- Gestalt beim Durchschlagen (Q = 1.5 Q_s) -------------------------------------------------
r = laeufe[findfirst(==(1.5), fake)]
reale = (sys.nmax + 1):(sys.nmax + 1 + sys.N)
t_zeig = r.t_durch .+ [-0.6, -0.3, 0.0, 0.3, 0.6, 1.2] .* (r.t_durch / 3)
fig = with_theme(THEMA) do
    f = Figure(size=(620, 360))
    ax = Axis(f[1, 1]; xlabel=L"x/L", ylabel=L"z/L", aspect=DataAspect(),
              title="Durchschlagen bei Q = 1.5 Qₛ")
    grau = ["#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
    for (t, c) in zip(t_zeig, grau)
        k = argmin(abs.(r.h.time .- t))
        Y = elastica_lagen(sys, r.h.q[k], DELTA)
        lines!(ax, Y[1, reale] ./ L, Y[2, reale] ./ L; color=c, linewidth=2,
               label=@sprintf("t = %.2f T₁", r.h.time[k] / T1))
    end
    Legend(f[1, 2], ax)
    f
end
speichern(fig, joinpath(OUT, "02_gestalt"))

##-- Energiebilanz (Q = 1.5 Q_s) ------------------------------------------------------------
fig = with_theme(THEMA) do
    f = Figure(size=(620, 320))
    ax = Axis(f[1, 1]; xlabel=L"t/T_1", ylabel=L"E\;[\mathrm{J}]", title="Energie bei Q = 1.5 Qₛ")
    t = r.h.time ./ T1
    lines!(ax, t, r.h.W .+ r.h.Ekin .- r.h.Wext; color=REF_FARBE, label="gesamt")
    lines!(ax, t, r.h.Ekin; color=FARBEN["B"], linestyle=:dash, linewidth=1.5, label="kinetisch")
    lines!(ax, t, r.h.W; color=FARBEN["B"], linestyle=:dot, linewidth=1.5, label="Formänderung")
    lines!(ax, t, -r.h.Wext; color=FARBEN["A"], linewidth=1.5, label="Potential der Last")
    Legend(f[1, 2], ax)
    f
end
speichern(fig, joinpath(OUT, "03_energie"))
Eg = r.h.W .+ r.h.Ekin .- r.h.Wext
@printf("Energiefehler (Q = 1.5 Q_s): max|ΔE|/max W = %.1e\n",
        maximum(abs.(Eg .- Eg[1])) / maximum(r.h.W))

##-- Schwelle: Durchschlagzeit über Q ----------------------------------------------------------
fschwelle = 1.0:0.05:1.6
ts = [lauf(sys, q0, f, Qs).t_durch for f in fschwelle]
println("Durchschlagzeit t/T₁ über Q/Q_s:")
for (f, t) in zip(fschwelle, ts)
    @printf("  %.2f: %s\n", f, isnan(t) ? "-" : @sprintf("%.2f", t / T1))
end

##-- Konvergenz der Dynamik in N (Q = 1.5 Q_s) -------------------------------------------------
fig = with_theme(THEMA) do
    f = Figure(size=(620, 320))
    ax = Axis(f[1, 1]; xlabel=L"t/T_1", ylabel=L"z_\mathrm{mitte}/L", title="Q = 1.5 Qₛ")
    for (N, c) in zip((50, 100, 200), ["#9ec5f4", "#2a78d6", "#0d366b"])
        s, q = aufbau(N)
        Qn = durchschlagkraft(s, q)
        rn = lauf(s, q, 1.5, Qn; tend=0.05)
        lines!(ax, rn.h.time ./ T1, rn.h.zmitte ./ L; color=c, linewidth=1.5, label="N = $N")
        @printf("N = %3d: Q_s = %.4f F_cr, Durchschlag bei t = %.4f s, Δt = %.2e s\n", N, Qn,
                rn.t_durch, rn.Δt)
    end
    axislegend(ax; position=:rt)
    f
end
speichern(fig, joinpath(OUT, "04_konvergenz"))
println("\nDiagramme in ", abspath(OUT))
