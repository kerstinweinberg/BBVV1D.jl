# Referenzlösungen für den 1D-Stab

# Geschwindigkeitspuls (volle Sinusperiode) und seine Verschiebung g(t) = ∫ v dt
puls_v(t, vmax, T) = 0 ≤ t < T ? vmax * sin(2π / T * t) : 0.0
puls_u(t, vmax, T) = 0 ≤ t < T ? vmax * T / 2π * (1 - cos(2π / T * t)) : 0.0

"""
    stab_analytisch(f, x, t; c, x0, l)

d'Alembert-Lösung im Stab [x0, l]: bei x0 ist f(t) vorgegeben (u oder v), das Ende l ist frei.
Spiegelungsreihe: Σ (-1)^n [f(t - (2nL + s)/c) + f(t - (2(n+1)L - s)/c)], s = x - x0, L = l - x0.
"""
function stab_analytisch(f, x, t; c, x0, l)
    L = l - x0
    s = x - x0
    val = 0.0
    n = 0
    while 2n * L / c ≤ t
        val += (-1)^n * (f(t - (2n * L + s) / c) + f(t - (2(n + 1) * L - s) / c))
        n += 1
    end
    return val
end

# Bondabstände ξ = ±nΔx im Inneren (|ξ| ≤ δ)
function nachbar_abstaende(Δx, δ)
    nmax = floor(Int, δ / Δx + 1e-12)
    return [n * Δx for n in -nmax:nmax if n != 0]
end

"""
    dispersion(model, k; Δx, δ, E, α=E, kernel=linear_kernel)

ρω² = Φ(k) im Inneren für die diskret kalibrierten Modelle (BB mit Energiekorrektur, CC
diskret, OSB mit Kernel ω), Formeln wie in docs/modelle.tex. Rückgabe ω/(c k) = √(Φ/E)/k.
"""
function dispersion(model, k; Δx, δ, E, α=E, kernel=linear_kernel)
    ξ = nachbar_abstaende(Δx, δ)
    L = abs.(ξ)
    V = Δx
    if model === :BB
        q = sum(L) * V
        Φ = 2E / q * sum((1 .- cos.(k .* ξ)) ./ L) * V
    elseif model === :CC
        q, M3 = sum(L) * V, sum(L .^ 3) * V
        Pp = sum((1 .+ cos.(k .* ξ)) .* L) * V
        Pm = sum((1 .- cos.(k .* ξ)) .* L) * V
        Φ = E / (q * M3) * Pp * Pm
    elseif model === :OSB
        ω = kernel.(δ, L)
        m = sum(ω .* ξ .^ 2) * V
        S = sum(ω .* ξ .* sin.(k .* ξ)) * V
        C = sum(ω .* (1 .- cos.(k .* ξ))) * V
        Φ = E * S^2 / m^2 + α / m * (2C - S^2 / m)
    else
        throw(ArgumentError("unknown model $model"))
    end
    return sqrt(max(Φ, 0.0) / E) / k
end
