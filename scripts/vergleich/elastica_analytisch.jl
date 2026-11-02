# Elastica des beidseitig gelenkig gelagerten Druckstabes (Euler-Fall II), Länge L.
# Mit k = √(F/EI) und p = sin(φ₀/2) (φ₀ Endwinkel):
#   kL = 2K(p),  F/F_cr = (2K/π)²,  Sehne c/L = 2E/K - 1,  w_max/L = p/K.
# Gestalt über ψ ∈ [-π/2, π/2] mit sin(φ/2) = p sin ψ:
#   s = (F(ψ) + K)/k,  x = (2E(ψ) - F(ψ) + 2E - K)/k,  w = (2p/k) cos ψ.

"vollständige elliptische Integrale K(p), E(p) (Modul p) über das AGM"
function ellKE(p)
    a, b, c = 1.0, sqrt(1 - p^2), p
    s = 0.5 * c^2
    fak = 0.5
    while abs(c) > 1e-16
        a, b, c = (a + b) / 2, sqrt(a * b), (a - b) / 2
        fak *= 2
        s += fak * c^2
    end
    K = π / (2a)
    return K, K * (1 - s)
end

"unvollständige Integrale F(ψ, p), E(ψ, p) durch Simpson-Quadratur"
function ellFE(ψ, p; n=2000)
    h = ψ / n
    f(t) = 1 / sqrt(1 - p^2 * sin(t)^2)
    e(t) = sqrt(1 - p^2 * sin(t)^2)
    simpson(g) = h / 3 * (g(0) + g(ψ) + 4sum(g((2i - 1) * h) for i in 1:n÷2) +
                          2sum(g(2i * h) for i in 1:n÷2-1))
    return simpson(f), simpson(e)
end

"exakte Lösung zur Sehne c (0 < … ≤ L): p, F/F_cr, w_max/L"
function elastica_zur_sehne(c_durch_L)
    sehne(p) = (KE = ellKE(p); 2KE[2] / KE[1] - 1)
    a, b = 0.0, 1 - 1e-12                       # sehne fällt monoton von 1 auf -1
    for _ in 1:200
        m = (a + b) / 2
        sehne(m) > c_durch_L ? (a = m) : (b = m)
    end
    p = (a + b) / 2
    K, _ = ellKE(p)
    return (; p, F_Fcr=(2K / π)^2, wmax_L=p / K)
end

"Gestalt (x, w) der exakten Elastica zum Parameter p, Länge L"
function elastica_gestalt(p, L; n=201)
    K, E = ellKE(p)
    k = 2K / L
    ψs = range(-π / 2, π / 2; length=n)
    x = similar(collect(ψs)); w = similar(x)
    for (i, ψ) in enumerate(ψs)
        Fψ, Eψ = ellFE(ψ, p)
        x[i] = (2Eψ - Fψ + 2E - K) / k
        w[i] = 2p / k * cos(ψ)
    end
    return x, w
end
