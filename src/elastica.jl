# Peridynamische Elastica: ebener Stab mit großen Verschiebungen und Drehungen.
#
# Unbekannt ist die Lage y_i = (x_i, z_i) jedes Knotens. Die Energie hat zwei Anteile:
#   Biegung (Modell B mit Winkelkrümmung): für jedes Nachbarpaar (i-n, i+n) der Winkel φ
#     zwischen den Bindungen a = y_i - y_{i-n} und b = y_{i+n} - y_i, κ_n = φ/(nΔs),
#     Ψ_i = ½EI ⟨κ_n²⟩ -- exakt drehinvariant, für kleine Verformungen das lineare Modell B;
#   Dehnung (BB): ε_ij = (|y_j - y_i| - L)/L, Ψ_i = ½EA Σ_j ε² L V_j / M₁ⁱ.
# W = Σ V_i Ψ_i. Gelenkige Lager über punktgespiegelte Geisterknoten y_{-n} = 2y_0 - y_n,
# damit ist die Krümmung am Lager null.

"""
    ElasticaSystem(L, N, δ, EI, EA)

Gerader Stab der Länge `L` mit Knoten X_i = iΔs, i = 0..N, beidseitig gelenkig gelagert:
y_0 = (0, 0) fest, y_N = (L - Δ, 0) mit vorgegebener Verkürzung Δ (Wegsteuerung).
Freiheitsgrade: y_1 … y_{N-1}, als Vektor q = [x_1, z_1, x_2, z_2, …].
"""
struct ElasticaSystem
    L::Float64
    N::Int
    Δs::Float64
    δ::Float64
    EI::Float64
    EA::Float64
    nmax::Int
    volume::Vector{Float64}   # reale Knoten 0..N
    M1::Vector{Float64}       # Σ_j L V_j je realem Knoten (mit Geisterknoten)
end

function ElasticaSystem(L::Real, N::Int, δ::Real, EI::Real, EA::Real)
    Δs = L / N
    nmax = floor(Int, δ / Δs + 1e-12)
    volume = fill(Δs, N + 1)
    volume[1] = volume[end] = Δs / 2
    # Geisterknoten haben volles Volumen Δs, deshalb ist M₁ an jedem Knoten vollständig
    M1 = fill(sum(2n * Δs * Δs for n in 1:nmax), N + 1)
    return ElasticaSystem(Float64(L), N, Δs, Float64(δ), Float64(EI), Float64(EA), nmax,
                          volume, M1)
end

# Spalte des Knotens i (i = -nmax .. N+nmax) in der vollen Lagematrix Y (2 × (N+1+2nmax))
@inline spalte(sys::ElasticaSystem, i) = i + sys.nmax + 1

"volle Lagematrix aus den Freiheitsgraden q und der Verkürzung Δ"
function elastica_lagen(sys::ElasticaSystem, q::AbstractVector, Δ::Real)
    (; N, nmax) = sys
    Y = zeros(eltype(q), 2, N + 1 + 2nmax)
    Y[:, spalte(sys, 0)] .= 0
    Y[1, spalte(sys, N)] = sys.L - Δ
    for i in 1:N-1
        Y[1, spalte(sys, i)] = q[2i - 1]
        Y[2, spalte(sys, i)] = q[2i]
    end
    for n in 1:nmax
        Y[:, spalte(sys, -n)] .= 2 .* Y[:, spalte(sys, 0)] .- Y[:, spalte(sys, n)]
        Y[:, spalte(sys, N + n)] .= 2 .* Y[:, spalte(sys, N)] .- Y[:, spalte(sys, N - n)]
    end
    return Y
end

# Drehwinkel von a nach b, im Bereich (-π, π]
@inline winkel(a1, a2, b1, b2) = atan(a1 * b2 - a2 * b1, a1 * b1 + a2 * b2)

"""
    elastica_energie(sys, Y) -> W
    elastica_energie!(G, sys, Y) -> W, G = ∂W/∂Y (alle Knoten, auch Geister)
"""
elastica_energie(sys::ElasticaSystem, Y::AbstractMatrix) =
    elastica_energie!(nothing, sys, Y)

function elastica_energie!(G, sys::ElasticaSystem, Y::AbstractMatrix)
    (; N, nmax, Δs, EI, EA, volume, M1) = sys
    G === nothing || fill!(G, 0)
    W = 0.0
    for i in 0:N
        ci = spalte(sys, i)
        Vi = volume[i + 1]
        # Biegung: Paare (i-n, i+n), alle vollständig wegen der Geisterknoten
        for n in 1:nmax
            cm, cp = spalte(sys, i - n), spalte(sys, i + n)
            a1, a2 = Y[1, ci] - Y[1, cm], Y[2, ci] - Y[2, cm]
            b1, b2 = Y[1, cp] - Y[1, ci], Y[2, cp] - Y[2, ci]
            φ = winkel(a1, a2, b1, b2)
            h = n * Δs
            W += Vi * 0.5 * EI / nmax * (φ / h)^2
            if G !== nothing
                f = Vi * EI / nmax * φ / h^2           # ∂W/∂φ
                aa, bb = a1^2 + a2^2, b1^2 + b2^2
                # ∂φ/∂b = (-b2, b1)/|b|², ∂φ/∂a = (a2, -a1)/|a|²
                gb1, gb2 = -b2 / bb, b1 / bb
                ga1, ga2 = a2 / aa, -a1 / aa
                G[1, cp] += f * gb1;  G[2, cp] += f * gb2
                G[1, ci] += f * (ga1 - gb1);  G[2, ci] += f * (ga2 - gb2)
                G[1, cm] -= f * ga1;  G[2, cm] -= f * ga2
            end
        end
        # Dehnung: alle Bindungen im Horizont
        for j in i-nmax:i+nmax
            j == i && continue
            cj = spalte(sys, j)
            L = abs(j - i) * Δs
            d1, d2 = Y[1, cj] - Y[1, ci], Y[2, cj] - Y[2, ci]
            l = sqrt(d1^2 + d2^2)
            ε = (l - L) / L
            W += Vi * 0.5 * EA * ε^2 * L * Δs / M1[i + 1]
            if G !== nothing
                f = Vi * EA * ε * Δs / M1[i + 1] / l   # ∂W/∂d = f d
                G[1, cj] += f * d1;  G[2, cj] += f * d2
                G[1, ci] -= f * d1;  G[2, ci] -= f * d2
            end
        end
    end
    return W
end

"""
    elastica_residuum(sys, q, Δ; Q=0) -> (r, F)

r = ∂Π/∂q mit Π = W - Q z_mitte (kleine Querkraft Q in der Mitte als Imperfektion) und die
Druckkraft F = -∂W/∂x_N am weggesteuerten Ende.
"""
function elastica_residuum(sys::ElasticaSystem, q::AbstractVector, Δ::Real; Q::Real=0.0)
    (; N, nmax) = sys
    Y = elastica_lagen(sys, q, Δ)
    G = similar(Y)
    elastica_energie!(G, sys, Y)
    # Geisterknoten auf die realen Knoten zurückführen (y_{-n} = 2y_0 - y_n usw.)
    Gr = G[:, spalte(sys, 0):spalte(sys, N)]
    for n in 1:nmax
        Gr[:, 1] .+= 2 .* G[:, spalte(sys, -n)]
        Gr[:, n + 1] .-= G[:, spalte(sys, -n)]
        Gr[:, N + 1] .+= 2 .* G[:, spalte(sys, N + n)]
        Gr[:, N - n + 1] .-= G[:, spalte(sys, N + n)]
    end
    r = vec(Gr[:, 2:N])
    r[2 * (N ÷ 2)] -= Q
    return r, -Gr[1, N + 1]
end

# Steifigkeitsmatrix ∂r/∂q durch gefärbte zentrale Differenzen (Kopplung nur über 2 nmax Knoten)
function elastica_tangente(sys::ElasticaSystem, q::AbstractVector, Δ::Real; Q::Real=0.0)
    n = length(q)
    reichweite = 2sys.nmax                        # Knotenabstand gekoppelter Freiheitsgrade
    nfarben = 2reichweite + 1
    h = 1e-7 * sys.L
    Ii, Jj, Vv = Int[], Int[], Float64[]
    for farbe in 1:nfarben, komp in 1:2
        knoten = farbe:nfarben:sys.N-1
        isempty(knoten) && continue
        e = zeros(n)
        for k in knoten
            e[2k - 2 + komp] = h
        end
        rp, _ = elastica_residuum(sys, q .+ e, Δ; Q)
        rm, _ = elastica_residuum(sys, q .- e, Δ; Q)
        dr = (rp .- rm) ./ (2h)
        for k in knoten, m in max(1, k - reichweite):min(sys.N - 1, k + reichweite), c in 1:2
            v = dr[2m - 2 + c]
            v == 0 && continue
            push!(Ii, 2m - 2 + c); push!(Jj, 2k - 2 + komp); push!(Vv, v)
        end
    end
    K = sparse(Ii, Jj, Vv, n, n)
    return (K + K') / 2
end

"Gesamtpotential Π = W - Q z_mitte"
function elastica_potential(sys::ElasticaSystem, q::AbstractVector, Δ::Real; Q::Real=0.0)
    return elastica_energie(sys, elastica_lagen(sys, q, Δ)) - Q * q[2 * (sys.N ÷ 2)]
end

"""
    elastica_gleichgewicht(sys, q0, Δ; Q=0, tol=1e-7, maxiter=50) -> (q, F, iter)

Minimum von Π(q) bei vorgegebener Verkürzung Δ: Newton-Verfahren, das nur abwärts geht.
Ist die Tangente nicht positiv definit (Startwert nahe einem instabilen Gleichgewicht, etwa
dem geraden gestauchten Stab), wird sie gedämpft (Levenberg), und die Schrittweite wird
über Π gesteuert. So wird ein stabiles Gleichgewicht gefunden.
"""
function elastica_gleichgewicht(sys::ElasticaSystem, q0::AbstractVector, Δ::Real;
                                Q::Real=0.0, tol::Real=1e-7, maxiter::Int=100)
    q = copy(q0)
    skala = sys.EI / sys.L^2                       # Kraftskala für das Abbruchkriterium
    r, F = elastica_residuum(sys, q, Δ; Q)
    Π = elastica_potential(sys, q, Δ; Q)
    for iter in 1:maxiter
        norm(r, Inf) < tol * skala && return q, F, iter - 1
        K = elastica_tangente(sys, q, Δ; Q)
        μ = 0.0
        dq = -(K \ r)
        while dot(dq, r) >= 0                       # keine Abstiegsrichtung -> dämpfen
            μ = μ == 0 ? 1e-6 * maximum(abs, diag(K)) : 10μ
            dq = -((K + μ * I) \ r)
        end
        # Rundungsgrenze: die Dehnsteifigkeit EA ≫ EI/L² begrenzt die erreichbare Genauigkeit
        norm(dq, Inf) < 1e-13 * sys.L && return q, F, iter
        α = 1.0
        while true
            qn = q .+ α .* dq
            Πn = elastica_potential(sys, qn, Δ; Q)
            if Πn <= Π + 1e-4 * α * dot(dq, r)
                q, Π = qn, Πn
                r, F = elastica_residuum(sys, q, Δ; Q)
                break
            elseif α < 1e-3
                # nahe am Minimum ändert sich Π nur noch im Rundungsbereich:
                # dann voller Newton-Schritt, wenn er das Residuum verkleinert
                qn = q .+ dq
                rn, Fn = elastica_residuum(sys, qn, Δ; Q)
                norm(rn) < norm(r) || return q, F, iter
                q, r, F = qn, rn, Fn
                Π = elastica_potential(sys, q, Δ; Q)
                break
            end
            α /= 2
        end
    end
    @warn "Newton nicht konvergiert" Δ norm(r, Inf) / skala
    return q, F, maxiter
end

"gerade Ausgangslage, gleichmäßig auf L - Δ gestaucht"
function elastica_gerade(sys::ElasticaSystem, Δ::Real=0.0)
    q = zeros(2(sys.N - 1))
    for i in 1:sys.N-1
        q[2i - 1] = i * sys.Δs * (1 - Δ / sys.L)
    end
    return q
end

"""
    elastica_knickform(sys, Δ)

Startwert auf dem geknickten Ast: erste Knickform z = A sin(πX/L) mit der Amplitude der
dehnstarren Näherung Δ ≈ (πA)²/(4L), also A = 2√(LΔ)/π.
"""
function elastica_knickform(sys::ElasticaSystem, Δ::Real)
    q = elastica_gerade(sys, Δ)
    A = 2 * sqrt(sys.L * Δ) / π
    for i in 1:sys.N-1
        q[2i] = A * sin(π * i / sys.N)
    end
    return q
end

"""
    elastica_pfad(sys, Δs; Q=0) -> Vektor von (; Δ, F, q, Y)

Nachknickpfad unter Wegsteuerung für die Verkürzungen `Δs` (aufsteigend, oberhalb der
Verzweigung). Startwert ist die Knickform zu `Δs[1]`, danach die Lösung des vorigen Schritts.
"""
function elastica_pfad(sys::ElasticaSystem, Δs::AbstractVector; Q::Real=0.0)
    q = elastica_knickform(sys, first(Δs))
    pfad = NamedTuple[]
    Δalt = first(Δs)
    for Δ in Δs
        # Startwert: x-Koordinaten linear mit dem Ende mitverschieben (auch für Sehne < 0)
        for i in 1:sys.N-1
            q[2i - 1] -= (Δ - Δalt) * i / sys.N
        end
        q, F, _ = elastica_gleichgewicht(sys, q, Δ; Q)
        push!(pfad, (; Δ, F, q=copy(q), Y=elastica_lagen(sys, q, Δ)))
        Δalt = Δ
    end
    return pfad
end
