# Referenzlösungen für den Bernoulli-Balken (c_B = √(EI/ρA), Kragarm: eingespannt bei x = 0)

"""
    dispersion_balken(modell, k; Δx, δ)

ω/(c_B k²) im Inneren: Modell A (gemittelte Krümmung, mit und ohne Randkorrektur gleich)
ρAω² = EI ĝ(k)², ĝ = (2/M₀) Σ (1 - cos kξ)/ξ² V; Modell B (Mittel der Quadrate)
ρAω² = EI ⟨κ̂_n²⟩, κ̂_n = 2(1 - cos k nΔx)/(nΔx)².
"""
function dispersion_balken(modell::Symbol, k; Δx, δ)
    nmax = floor(Int, δ / Δx + 1e-12)
    n = 1:nmax
    if modell === :A
        ĝ = 2 / (2nmax * Δx) * sum(2 .* (1 .- cos.(k .* n .* Δx)) ./ (n .* Δx) .^ 2) * Δx
        return ĝ / k^2
    elseif modell === :B
        κ̂ = 2 .* (1 .- cos.(k .* n .* Δx)) ./ (n .* Δx) .^ 2
        return sqrt(sum(κ̂ .^ 2) / nmax) / k^2
    end
    throw(ArgumentError("unknown model $modell"))
end

"Nullstellen von cos z cosh z + 1 = 0 (Kragarm), z_n ≈ (2n - 1)π/2"
function kragarm_wurzeln(n_moden)
    f(z) = cos(z) + 1 / cosh(z)
    return map(1:n_moden) do n
        a, b = (2n - 1) * π / 2 - 1, (2n - 1) * π / 2 + 1
        for _ in 1:100
            m = (a + b) / 2
            sign(f(m)) == sign(f(a)) ? (a = m) : (b = m)
        end
        (a + b) / 2
    end
end

"""
Eigenform des Kragarms φ(u), u = x/l, numerisch stabil auch für große z:
φ = cosh zu - cos zu - σ (sinh zu - sin zu), σ = (cosh z + cos z)/(sinh z + sin z).
"""
function kragarm_eigenform(z, u)
    q = 2 / (1 - exp(-2z) + 2sin(z) * exp(-z))            # e^z / (sinh z + sin z)
    einsminussigma = (-exp(-z) + sin(z) - cos(z)) * q * exp(-z)
    σ = 1 - einsminussigma
    y = z * u
    # cosh y - σ sinh y = ½[(1-σ) e^y + (1+σ) e^-y], (1-σ) e^y ohne Überlauf
    return 0.5 * ((-exp(-z) + sin(z) - cos(z)) * q * exp(y - z) + (1 + σ) * exp(-y)) -
           cos(y) + σ * sin(y)
end

"""
    kragarm_loslassen(; l, c, κ0, n_moden=40) -> w(x, t)

Kragarm, eingespannt bei x = 0, durch ein Endmoment kreisförmig gebogen (w = κ₀x²/2) und bei
t = 0 losgelassen. Modalreihe w = Σ aₙ φₙ(x/l) cos ωₙt, ωₙ = (zₙ/l)² c_B.
"""
function kragarm_loslassen(; l, c, κ0, n_moden=40)
    z = kragarm_wurzeln(n_moden)
    u = range(0, 1; length=4001)
    simpson(f) = (h = step(u); h / 3 * (f[1] + f[end] + 4sum(f[2:2:end-1]) + 2sum(f[3:2:end-2])))
    w0 = κ0 .* (u .* l) .^ 2 ./ 2
    a = map(z) do zn
        φ = kragarm_eigenform.(zn, u)
        simpson(w0 .* φ) / simpson(φ .^ 2)
    end
    ω = (z ./ l) .^ 2 .* c
    return (x, t) -> sum(a[n] * kragarm_eigenform(z[n], x / l) * cos(ω[n] * t) for n in 1:n_moden)
end
