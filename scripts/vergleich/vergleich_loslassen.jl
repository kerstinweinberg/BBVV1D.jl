# Plötzliches Loslassen eines gekrümmten Kragarms (Spaghetti-Mechanismus, Audoly & Neukirch):
# Kreisbogen mit Krümmung κ₀, eingespannt bei s = 0, bei t = 0 am Ende s = L losgelassen.
# Linear schwingt die Krümmung selbstähnlich auf 1.428 κ₀ über (η = x/√(c_B t) = 2√π, x vom
# freien Ende). Hier: peridynamische Elastica für schwach bis stark gekrümmte Stäbe.
#
#   julia --project=scripts/vergleich scripts/vergleich/vergleich_loslassen.jl [ausgabepfad]

using BBVV1D, LinearAlgebra, Printf
include("plotting.jl")

OUT::String = length(ARGS) ≥ 1 ? ARGS[1] :
              joinpath(@__DIR__, "..", "..", "results", "vergleich", "loslassen")

# Knickstab-Querschnitt wie in knickstab.jl, SI-Einheiten
const E = 3.375e9
const RHO = 1500.0
const D = 1.66e-3
const L = 0.254
const EI = E * π * D^4 / 64
const EA = E * π * D^2 / 4
const RHOA = RHO * π * D^2 / 4
const CB = sqrt(EI / RHOA)                 # m²/s
const TEND = (L / 16)^2 / CB               # η = 16 bei der Einspannung: diese spürt noch nichts
const N0 = 800
# Loslassen über τ_r statt schlagartig: das Endmoment EIκ₀ wird linear weggenommen. Schlagartig
# regt der Krümmungssprung Wellen bis zur Gitterweite an, die im Bernoulli-Modell beliebig
# schnell, in der PD aber zu langsam sind und bei Verfeinerung zunehmen; τ_r filtert
# Wellenlängen unter √(c_B τ_r) ≈ 4 mm, weit unter der Breite des Überschwingens.
const TAU = 25e-6
endmoment(κ0) = t -> EI * κ0 * max(0.0, 1 - t / TAU)

printstyled(@sprintf("Kragarm L = %.0f mm, d = %.2f mm, c_B = %.4f m²/s, t_end = %.3e s\n",
                     1e3L, 1e3D, CB, TEND); color=:blue, bold=true)

"""
lineare Ähnlichkeitslösung (knickstabWelle.jl, Fresnel), x vom freien Ende:
κ/κ₀ = f(η) = √(2/π) ∫₀^η sin(s²/4) ds, η = x/√(c_B t); Maximum 1.4279 bei η = 2√π
"""
function faehnlich(η; n=2000)
    h = η / n
    g(s) = sin(s^2 / 4)
    return sqrt(2 / π) * h / 3 * (g(0) + g(η) + 4sum(g((2i - 1) * h) for i in 1:n÷2) +
                                  2sum(g(2i * h) for i in 1:n÷2-1))
end
ref(x, t) = faehnlich(x / sqrt(CB * t))

function zeitschritt(sys, q0)
    M = repeat(RHOA .* sys.volume[2:end]; inner=2)
    h = 1e-9 * L
    v = randn(length(q0))
    λ = 0.0
    for _ in 1:40
        Kv = (kragarm_gradient(sys, q0 .+ h .* v) .- kragarm_gradient(sys, q0 .- h .* v)) ./ (2h)
        u = Kv ./ M
        λ = norm(u) / norm(v)
        v = u ./ norm(u)
    end
    return 0.5 * 2 / sqrt(λ)
end

function loslassen(κ0L; N=N0, n_out=80)
    sys = ElasticaSystem(L, N, 3.015L / N, EI, EA; rechts_frei=true)
    κ0 = κ0L / L
    q0 = kragarm_bogen(sys, κ0)
    r = kragarm_dynamik(sys, RHOA; q0, tend=TEND, Δt=zeitschritt(sys, q0), n_out,
                        endmoment=endmoment(κ0))
    xfrei = L .- (0:N) .* sys.Δs                      # Abstand vom freien Ende
    # Überschwingen: Maximum von κ/κ₀ im ersten Bogen η ∈ [1, 5] (das nächste Minimum der
    # linearen Lösung liegt bei 2√(2π) = 5.01); außerhalb liegen Randschicht und Fernfeld
    bogen(t) = findall(x -> 1 <= x / sqrt(CB * t) <= 5, xfrei)
    ueber = [t == 0 ? 1.0 : maximum(k[bogen(t)] ./ κ0) for (k, t) in zip(r.κ, r.time)]
    lage = [t == 0 ? NaN : xfrei[bogen(t)][argmax(k[bogen(t)])] for (k, t) in zip(r.κ, r.time)]
    # Energie bleibt erst nach dem Loslassen (t > τ_r) erhalten, vorher leistet das Moment Arbeit
    nach = findall(>(TAU), r.time)
    E0 = r.W[nach[1]] + r.Ekin[nach[1]]
    dE = maximum(abs.(r.W[nach] .+ r.Ekin[nach] .- E0)) / E0
    return (; sys, κ0, κ0L, r, xfrei, ueber, lage, dE)
end

##-- Läufe ----------------------------------------------------------------------------------
κ0Ls = [0.01, 4π, 8π, 16π]                   # θ_end = κ₀L/16 = 0, π/4, π/2, π
laeufe = [loslassen(k) for k in κ0Ls]
for l in laeufe
    @printf("κ₀L = %6.3f: Energiefehler %.1e, max κ/κ₀ am Ende = %.4f\n", l.κ0L, l.dE, l.ueber[end])
end

##-- 1. linearer Fall: Krümmungsprofile über η gegen die Modalreihe ------------------------
lin = laeufe[1]
ts = [0.1, 0.25, 0.5, 1.0] .* TEND
fig = with_theme(THEMA) do
    f = Figure(size=(620, 340))
    ax = Axis(f[1, 1]; xlabel=L"\eta = x/\sqrt{c_B t}", ylabel=L"\kappa/\kappa_0",
              title="schwach gekrümmt (κ₀L = 0.01), N = $N0")
    ηr = range(0, 8; length=400)
    lines!(ax, ηr, faehnlich.(ηr); color=REF_FARBE,
           linestyle=:dash, linewidth=1.5, label="linear (Fresnel)")
    for (t, c) in zip(ts, ["#9ec5f4", "#6da7ec", "#2a78d6", "#0d366b"])
        k = argmin(abs.(lin.r.time .- t))
        η = lin.xfrei ./ sqrt(CB * lin.r.time[k])
        sel = η .<= 8
        lines!(ax, η[sel], lin.r.κ[k][sel] ./ lin.κ0; color=c, linewidth=1.5,
               label=@sprintf("PD, t = %.2f t_end", lin.r.time[k] / TEND))
    end
    scatter!(ax, [2sqrt(π)], [1.42794]; color=REF_FARBE, markersize=8)
    Legend(f[1, 2], ax)
    f
end
speichern(fig, joinpath(OUT, "01_profil_linear"))
kr = argmin(abs.(lin.r.time .- TEND))
fehler_lin(η1, η2) = maximum(abs(lin.r.κ[kr][i] / lin.κ0 - ref(lin.xfrei[i], lin.r.time[kr]))
                             for i in eachindex(lin.xfrei)
                             if η1 <= lin.xfrei[i] / sqrt(CB * TEND) <= η2)
@printf("linear, t_end: max|κ_PD - κ_ref|/κ₀ = %.1e (η ∈ [0.5, 5]), %.1e (η ∈ [5, 8]), %.1e (η < 0.5)\n",
        fehler_lin(0.5, 5), fehler_lin(5, 8), fehler_lin(0, 0.5))
@printf("linear: Überschwingen PD %.4f (exakt 1.4279)\n", lin.ueber[end])

##-- 2. Überschwingen über θ = κ₀√(c_B t) ----------------------------------------------------
farben = ["#3d3d3a", "#9ec5f4", "#6da7ec", "#2a78d6", "#0d366b"]
fig = with_theme(THEMA) do
    f = Figure(size=(620, 340))
    ax = Axis(f[1, 1]; xlabel=L"\theta = \kappa_0\sqrt{c_B t}", ylabel=L"\max_x \kappa/\kappa_0",
              title="Überschwingen der Krümmung")
    hlines!(ax, 1.42794; color=REF_FARBE, linestyle=:dash, linewidth=1.5, label="linear 1.428")
    for (l, c) in zip(laeufe[2:end], farben[2:end])
        θ = l.κ0 .* sqrt.(CB .* l.r.time)
        sel = l.r.time .>= 0.05TEND                # zu früh: Profil schmaler als δ
        lines!(ax, θ[sel], l.ueber[sel]; color=c, linewidth=2,
               label=@sprintf("κ₀L = %.0fπ", l.κ0L / π))
    end
    Legend(f[1, 2], ax)
    f
end
speichern(fig, joinpath(OUT, "02_ueberschwingen"))
println("Überschwingen über θ (Ende des Laufs):")
for l in laeufe
    @printf("  κ₀L = %6.3f: θ_end = %.3f, max κ/κ₀ = %.4f bei η = %.2f\n", l.κ0L,
            l.κ0 * sqrt(CB * TEND), l.ueber[end], l.lage[end] / sqrt(CB * TEND))
end

##-- 3. Gestalt des stark gekrümmten Stabes --------------------------------------------------
sg = laeufe[findfirst(==(4π), κ0Ls)]
fig = with_theme(THEMA) do
    f = Figure(size=(620, 460))
    ax = Axis(f[1, 1]; xlabel=L"x\;[\mathrm{m}]", ylabel=L"z\;[\mathrm{m}]", aspect=DataAspect(),
              title="Loslassen bei κ₀L = 4π, freie Hälfte (eine Windung)")
    # nur die freie Hälfte s ≥ L/2; die Einspannung liegt eine Windung weiter innen
    haelfte = (sg.sys.nmax + 1 + sg.sys.N ÷ 2):(sg.sys.nmax + 1 + sg.sys.N)
    for (t, c) in zip([0.0, 0.25, 0.5, 1.0] .* TEND, ["#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
        k = argmin(abs.(sg.r.time .- t))
        Y = kragarm_lagen(sg.sys, sg.r.q[k])
        lines!(ax, Y[1, haelfte], Y[2, haelfte]; color=c, linewidth=2,
               label=@sprintf("t = %.2f t_end", sg.r.time[k] / TEND))
        scatter!(ax, [Y[1, haelfte[end]]], [Y[2, haelfte[end]]]; color=c, markersize=8)
    end
    Legend(f[1, 2], ax)
    f
end
speichern(fig, joinpath(OUT, "03_gestalt"))
st = laeufe[findfirst(==(8π), κ0Ls)]

##-- 4. Krümmungsprofile des stark gekrümmten Stabes -----------------------------------------
fig = with_theme(THEMA) do
    f = Figure(size=(620, 340))
    ax = Axis(f[1, 1]; xlabel=L"\eta = x/\sqrt{c_B t}", ylabel=L"\kappa/\kappa_0",
              title="stark gekrümmt (κ₀L = 8π)")
    ηr = range(0, 8; length=400)
    lines!(ax, ηr, faehnlich.(ηr); color=REF_FARBE,
           linestyle=:dash, linewidth=1.5, label="linear")
    for (t, c) in zip(ts, ["#9ec5f4", "#6da7ec", "#2a78d6", "#0d366b"])
        k = argmin(abs.(st.r.time .- t))
        η = st.xfrei ./ sqrt(CB * st.r.time[k])
        sel = η .<= 8
        lines!(ax, η[sel], st.r.κ[k][sel] ./ st.κ0; color=c, linewidth=1.5,
               label=@sprintf("θ = %.2f", st.κ0 * sqrt(CB * st.r.time[k])))
    end
    Legend(f[1, 2], ax)
    f
end
speichern(fig, joinpath(OUT, "04_profil_stark"))

##-- 5. Konvergenz in N (linear, Überschwingen bei t_end) -------------------------------------
println("Konvergenz Überschwingen (κ₀L = 0.01 und 8π, t_end):")
for N in (200, 400, 800, 1600)
    a = loslassen(0.01; N, n_out=4)
    b = loslassen(8π; N, n_out=4)
    @printf("  N = %3d: linear %.4f   κ₀L = 8π: %.4f\n", N, a.ueber[end], b.ueber[end])
end
println("\nDiagramme in ", abspath(OUT))

##-- 6. Gestalt des Profils über θ: erstes Maximum und folgende Senke --------------------------
"erstes Maximum (Wert, η) im Bogen η ∈ [1, 5] und das Minimum danach bis η + 3"
function profilkennwerte(l, k)
    t = l.r.time[k]
    η = l.xfrei ./ sqrt(CB * t)
    κ = l.r.κ[k] ./ l.κ0
    b = findall(e -> 1 <= e <= 5, η)
    i = b[argmax(κ[b])]
    d = findall(e -> η[i] < e <= η[i] + 3, η)
    j = d[argmin(κ[d])]
    return (; θ=l.κ0 * sqrt(CB * t), kmax=κ[i], ηmax=η[i], kmin=κ[j], ηmin=η[j])
end
@printf("linear (Fresnel): Maximum %.4f bei η = %.3f, Senke %.4f bei η = %.3f\n",
        faehnlich(2sqrt(π)), 2sqrt(π), faehnlich(2sqrt(2π)), 2sqrt(2π))
println("θ       κ_max/κ₀  η_max   κ_min/κ₀  η_min   (κ₀L)")
for l in laeufe[2:end], frac in (0.25, 0.5, 1.0)
    k = argmin(abs.(l.r.time .- frac * TEND))
    p = profilkennwerte(l, k)
    @printf("%5.2f   %7.4f   %5.2f   %7.4f   %5.2f   %.0fπ\n", p.θ, p.kmax, p.ηmax, p.kmin, p.ηmin,
            l.κ0L / π)
end
