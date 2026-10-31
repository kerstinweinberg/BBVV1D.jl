# Peridynamischer Bernoulli-Balken (Durchbiegung w, kleine Verformungen).
#
# Die Modelle sind linear in w. Jedes Modell definiert eine diskrete Krümmung
# κ = G w_voll an Krümmungsstellen r mit Gewichten g_r, die Formänderungsarbeit ist
#     W = ½ EI Σ_r g_r κ_r² ,
# und mit der Abbildung w_voll = P w der Freiheitsgrade auf alle Knoten (Geisterknoten an
# den Lagern) folgt die Steifigkeitsmatrix H = EI Pᵀ Gᵀ diag(g) G P. Bewegungsgleichung
#     M ẅ = f - H w ,   M = diag(ρA V) ,
# also exakt variationell; Statik H w = f, Eigenfrequenzen H φ = ω² M φ.

#-- Modelle -------------------------------------------------------------------------------

abstract type AbstractBeamModel end

"""
    MittlereKruemmung(; randkorrektur=false)

Gemittelte Krümmung (CC-Modell, Notizen Kap. 66): κ̄_i = c_i Σ_j (w_j - w_i)/ξ² V_j mit
c_i = 2/Σ_j V_j, Energie ½EI κ̄². Mit `randkorrektur = true` werden die Gewichte mit einem
Reproducing Kernel (Basis ξ, ξ², Kernel ξ⁻⁴) so bestimmt, dass κ̄ für lineare und quadratische
Felder an jedem Punkt exakt ist; im Inneren sind beide Fassungen gleich.
"""
struct MittlereKruemmung <: AbstractBeamModel
    randkorrektur::Bool
end
MittlereKruemmung(; randkorrektur::Bool=false) = MittlereKruemmung(randkorrektur)

"""
    BindungsKruemmung()

Mittel der Quadrate der Paarkrümmung (BP-Modell, Bindungspaare): für jedes Nachbarpaar (i-n, i+n) im Horizont die Krümmung
κ_n = (w_{i+n} - 2w_i + w_{i-n})/(nΔx)², Energie ½EI ⟨κ_n²⟩ (gleiche Gewichte, normiert über
die vorhandenen Paare).
"""
struct BindungsKruemmung <: AbstractBeamModel end

#-- Gitter und Lager ----------------------------------------------------------------------

"""
    BeamSystem(l, N, δ, EI, ρA, modell; links=:eingespannt, rechts=:frei)

Knoten x_i = iΔx, i = 0..N, Δx = l/N, Volumen Δx (Endknoten Δx/2). Lager je Ende:
`:frei`, `:gelenkig` (w = 0, Geisterknoten antisymmetrisch, also w'' = 0) oder
`:eingespannt` (w = 0, Geisterknoten symmetrisch, also w' = 0).
Felder: `x` (Knoten), `dofs` (freie Knoten), `H`, `M`, `G`, `g`, `P`.
"""
struct BeamSystem{T<:AbstractBeamModel}
    modell::T
    l::Float64
    Δx::Float64
    δ::Float64
    EI::Float64
    ρA::Float64
    x::Vector{Float64}
    volume::Vector{Float64}
    dofs::Vector{Int}
    G::SparseMatrixCSC{Float64,Int}
    g::Vector{Float64}
    P::SparseMatrixCSC{Float64,Int}
    H::SparseMatrixCSC{Float64,Int}
    M::Vector{Float64}
end

function BeamSystem(l::Real, N::Int, δ::Real, EI::Real, ρA::Real, modell::AbstractBeamModel;
                    links::Symbol=:eingespannt, rechts::Symbol=:frei)
    for lager in (links, rechts)
        lager in (:frei, :gelenkig, :eingespannt) ||
            throw(ArgumentError("Lager $lager unbekannt, :frei, :gelenkig oder :eingespannt"))
    end
    Δx = l / N
    ng = floor(Int, δ / Δx + 1e-12)
    x = collect(0:N) .* Δx
    volume = fill(Δx, N + 1)
    volume[1] = volume[end] = Δx / 2
    # alle Knoten: Geister links (Index -ng..-1), reale 0..N, Geister rechts N+1..N+ng
    lo = links === :frei ? 0 : -ng
    hi = rechts === :frei ? N : N + ng
    knoten = collect(lo:hi)
    xv = knoten .* Δx
    nv = length(knoten)
    pos(i) = i - lo + 1                     # Knotenindex -> Spalte in w_voll
    fest = Int[]
    links === :frei || push!(fest, 0)
    rechts === :frei || push!(fest, N)
    dofs = [i for i in 0:N if !(i in fest)]
    dofpos = Dict(i => k for (k, i) in enumerate(dofs))
    # P: w_voll = P w_dof
    Pi, Pj, Pv = Int[], Int[], Float64[]
    for i in dofs
        push!(Pi, pos(i)); push!(Pj, dofpos[i]); push!(Pv, 1.0)
    end
    for n in 1:ng
        if links !== :frei && n in keys(dofpos)
            push!(Pi, pos(-n)); push!(Pj, dofpos[n]); push!(Pv, links === :gelenkig ? -1.0 : 1.0)
        end
        if rechts !== :frei && (N - n) in keys(dofpos)
            push!(Pi, pos(N + n)); push!(Pj, dofpos[N - n])
            push!(Pv, rechts === :gelenkig ? -1.0 : 1.0)
        end
    end
    P = sparse(Pi, Pj, Pv, nv, length(dofs))
    G, g = kruemmungsoperator(modell, knoten, xv, Δx, δ, N, volume)
    H = EI * (P' * G' * Diagonal(g) * G * P)
    H = (H + H') / 2
    M = ρA .* volume[dofs .+ 1]
    return BeamSystem(modell, Float64(l), Δx, Float64(δ), Float64(EI), Float64(ρA), x,
                      volume, dofs, G, g, P, sparse(H), M)
end

# Krümmung an jedem realen Knoten (Zeile i+1), Gewicht = Knotenvolumen
function kruemmungsoperator(m::MittlereKruemmung, knoten, xv, Δx, δ, N, volume)
    Gi, Gj, Gv = Int[], Int[], Float64[]
    for i in 0:N
        ci = findfirst(==(i), knoten)
        nb = [k for k in eachindex(knoten) if 0 < abs(xv[k] - xv[ci]) <= δ + 1e-12 * Δx]
        ξ = xv[nb] .- xv[ci]
        if m.randkorrektur
            # Reproducing Kernel, Basis Q = (ξ, ξ²), Kernel ξ⁻⁴: Σ a ξ = 0, Σ a ξ² = 2
            Mm = zeros(2, 2)
            for s in ξ
                Q = [s, s^2]
                Mm .+= (Q * Q') .* (Δx / s^4)
            end
            b = Mm \ [0.0, 2.0]
            a = [Δx / s^4 * (s * b[1] + s^2 * b[2]) for s in ξ]
        else
            c = 2 / (length(ξ) * Δx)
            a = [c * Δx / s^2 for s in ξ]
        end
        for (k, ak) in zip(nb, a)
            push!(Gi, i + 1); push!(Gj, k); push!(Gv, ak)
            push!(Gi, i + 1); push!(Gj, ci); push!(Gv, -ak)
        end
    end
    return sparse(Gi, Gj, Gv, N + 1, length(knoten)), copy(volume)
end

# eine Krümmung je Nachbarpaar (i-n, i+n), Gewicht V_i / Anzahl der Paare
function kruemmungsoperator(::BindungsKruemmung, knoten, xv, Δx, δ, N, volume)
    nmax = floor(Int, δ / Δx + 1e-12)
    Gi, Gj, Gv = Int[], Int[], Float64[]
    g = Float64[]
    r = 0
    for i in 0:N
        ci = findfirst(==(i), knoten)
        paare = [n for n in 1:nmax if ci - n >= 1 && ci + n <= length(knoten)]
        for n in paare
            r += 1
            h2 = (n * Δx)^2
            append!(Gi, (r, r, r)); append!(Gj, (ci - n, ci, ci + n))
            append!(Gv, (1 / h2, -2 / h2, 1 / h2))
            push!(g, volume[i + 1] / length(paare))
        end
    end
    return sparse(Gi, Gj, Gv, r, length(knoten)), g
end

#-- Auswertung ----------------------------------------------------------------------------

"Verschiebung an allen realen Knoten 0..N (feste Lager w = 0)"
function knotenwerte(sys::BeamSystem, w::AbstractVector)
    v = zeros(length(sys.x))
    v[sys.dofs .+ 1] .= w
    return v
end

"Krümmung des Modells an den Krümmungsstellen (CC: an den Knoten)"
kruemmung(sys::BeamSystem, w::AbstractVector) = sys.G * (sys.P * w)

"Formänderungsarbeit W = ½ wᵀ H w"
formaenderungsarbeit(sys::BeamSystem, w::AbstractVector) = 0.5 * dot(w, sys.H * w)

"Statik: H w = f mit Knotenkräften f (Streckenlast q: f = q V)"
statik(sys::BeamSystem, f::AbstractVector) = Symmetric(Matrix(sys.H)) \ f

"""
    eigenfrequenzen(sys) -> (ω, Φ)

Kreisfrequenzen aufsteigend und Eigenformen (Spalten, an den Freiheitsgraden).
"""
function eigenfrequenzen(sys::BeamSystem)
    s = 1 ./ sqrt.(sys.M)
    A = Symmetric(Diagonal(s) * Matrix(sys.H) * Diagonal(s))
    λ, V = eigen(A)
    ω = sqrt.(max.(λ, 0.0))
    return ω, Diagonal(s) * V
end

# Gershgorin-Schranke für ω²_max von M⁻¹H
function omega2_max(sys::BeamSystem)
    H = sys.H
    return maximum(sum(abs, H[:, k]) / sys.M[k] for k in axes(H, 2))
end

"stabiler Zeitschritt des Velocity-Verlet-Verfahrens, Δt = safety · 2/ω_max"
stabiler_zeitschritt(sys::BeamSystem; safety_factor::Real=0.7) =
    safety_factor * 2 / sqrt(omega2_max(sys))

"""
    balken_dynamik(sys; w0, v0=0, f=t->0, tend, Δt=stabiler_zeitschritt(sys), n_out=200)

Velocity-Verlet für M ẅ = f(t) - H w. Rückgabe `(; time, position, displacement, velocity)`
an allen realen Knoten, Matrizen (N+1) × n_out, wie `simulation(...; history=true)`.
"""
function balken_dynamik(sys::BeamSystem; w0::AbstractVector, v0=zero(w0), f=t -> 0.0,
                        tend::Real, Δt::Real=stabiler_zeitschritt(sys), n_out::Int=200)
    n_steps = ceil(Int, tend / Δt)
    Δt = tend / n_steps
    freq = max(1, n_steps ÷ n_out)
    w = copy(w0)
    v = collect(float.(v0 .+ zero(w0)))
    a = (f(0.0) .- sys.H * w) ./ sys.M
    time = [0.0]
    U = [knotenwerte(sys, w)]
    Vel = [knotenwerte(sys, v)]
    for step in 1:n_steps
        t = step * Δt
        @. v += 0.5Δt * a
        @. w += Δt * v
        a = (f(t) .- sys.H * w) ./ sys.M
        @. v += 0.5Δt * a
        if step % freq == 0
            push!(time, t); push!(U, knotenwerte(sys, w)); push!(Vel, knotenwerte(sys, v))
        end
    end
    return (; time, position=sys.x, displacement=reduce(hcat, U), velocity=reduce(hcat, Vel))
end
