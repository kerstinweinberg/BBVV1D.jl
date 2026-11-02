using BBVV1D
using Test

const L_STAB = 1.0
const ΔX = L_STAB / 100
const PARAMS = MaterialParameters(; horizon=3.015ΔX, rho=8e-6, E=2.1e5)
const PC = PointCloud(L_STAB, ΔX)
# Inneres: alle Nachbarn haben vollständige Familien
const INTERIOR = findall(x -> 2PARAMS.δ < x < L_STAB - 2PARAMS.δ, PC.position)

# b_int für ein Verschiebungsfeld u (klein, damit die geometrische Nichtlinearität
# der BB-/OSB-Formulierung vernachlässigbar ist)
function force(mat, u; params=PARAMS)
    system = BBVV1D.BondSystem(PC, mat, params)
    storage = BBVV1D.Storage(system, mat, params)
    storage.displacement .= u
    storage.position .= system.position .+ u
    BBVV1D.calc_force_density!(storage, system, mat, params)
    return copy(storage.b_int), storage, system
end

const MATS = ["BB" => BBMaterial(),
              "BB energy" => BBMaterial{EnergySurfaceCorrection}(),
              "CC continuum" => CCMaterial(),
              "CC discrete" => CCMaterial(; calibration=:discrete),
              "CCF" => CCFMaterial(),
              "BAQP" => BAQPMaterial(),
              "BAQP linear" => BAQPMaterial(; kernel=linear_kernel),
              "OSB const" => OSBMaterial(; kernel=const_one_kernel),
              "OSB linear" => OSBMaterial(),
              "OSB cubic" => OSBMaterial(; kernel=cubic_b_spline_kernel),
              "OSB linear energy" => OSBMaterial{EnergySurfaceCorrection}()]

# kalibrierte Modelle: b = E u'' im Inneren exakt
const CALIBRATED = ["BB energy", "CC discrete", "CCF", "BAQP", "BAQP linear", "OSB const",
                    "OSB linear", "OSB cubic", "OSB linear energy"]

@testset "BBVV1D.jl" begin
    @testset "homogeneous strain: $name" for (name, mat) in MATS
        ε = 1e-6
        b, _, _ = force(mat, ε .* PC.position)
        @test maximum(abs, b[INTERIOR]) < 1e-6 * PARAMS.E * ε / PARAMS.δ
    end

    @testset "calibration b = E u'': $name" for (name, mat) in MATS
        name in CALIBRATED || continue
        κ = 1e-6                                   # u'' = κ
        b, _, _ = force(mat, κ .* PC.position .^ 2 ./ 2)
        @test all(isapprox.(b[INTERIOR], PARAMS.E * κ; rtol=1e-4))
    end

    @testset "uncorrected constants" begin
        κ = 1e-6
        u = κ .* PC.position .^ 2 ./ 2
        b, _, _ = force(BBMaterial(), u)
        @test b[INTERIOR[1]] / (PARAMS.E * κ) ≈ 12 / 3.015^2 rtol = 1e-5   # q_diskret/δ²
        b, _, _ = force(CCMaterial(), u)
        @test b[INTERIOR[1]] / (PARAMS.E * κ) ≈ (1.5 / 3.015^3)^2 * 12 * 72 rtol = 1e-4
    end

    @testset "energy surface correction BB = E(1/q_i + 1/q_j)" begin
        mat = BBMaterial{EnergySurfaceCorrection}()
        system = BBVV1D.BondSystem(PC, mat, PARAMS)
        q = [sum(system.bonds[b].length * ΔX for b in BBVV1D.each_bond_idx(system, i))
             for i in BBVV1D.each_point_idx(system)]
        ok = true
        for i in BBVV1D.each_point_idx(system), bid in BBVV1D.each_bond_idx(system, i)
            j = system.bonds[bid].neighbor
            ok &= isapprox(system.scfactor[bid] * PARAMS.bc, PARAMS.E * (1/q[i] + 1/q[j]);
                           rtol=1e-3)
        end
        @test ok
        # OSB: θ ist auch am Rand exakt -> Korrekturfaktor 1
        system = BBVV1D.BondSystem(PC, OSBMaterial{EnergySurfaceCorrection}(), PARAMS)
        @test all(isapprox.(system.scfactor, 1; rtol=1e-3))
    end

    @testset "zigzag mode" begin
        u = [1e-6 * (-1)^i for i in 1:PC.n_points]
        b, _, _ = force(CCMaterial(; calibration=:discrete), u)
        @test minimum(abs, b[INTERIOR]) > 0      # CC: keine Nullenergie-Mode
        b, _, _ = force(OSBMaterial(), u)
        @test minimum(abs, b[INTERIOR]) > 0      # OSB mit α > 0: keine Nullenergie-Mode
        params0 = MaterialParameters(; horizon=3.015ΔX, rho=8e-6, E=2.1e5, alpha=0.0)
        b, _, _ = force(OSBMaterial(; kernel=const_one_kernel), u; params=params0)
        @test maximum(abs, b[INTERIOR]) < 1e-6 * maximum(abs, b)   # α = 0: energiefrei
        b, storage, _ = force(CCFMaterial(), u)
        @test maximum(abs, storage.barepsilon[INTERIOR]) < 1e-12   # CCF ≡ OSB, α = 0, ω = 1
        @test maximum(abs, b[INTERIOR]) < 1e-6 * maximum(abs, b)
    end

    @testset "CCF: barepsilon linear konsistent, b = -∂W/∂u / V" begin
        ε = 1e-6
        _, storage, _ = force(CCFMaterial(), ε .* PC.position)
        @test all(isapprox.(storage.barepsilon, ε; rtol=1e-10))   # auch am Rand
        u = 1e-6 .* randn(PC.n_points)
        W(u) = sum(0.5 * PARAMS.E * force(CCFMaterial(), u)[2].barepsilon .^ 2 .* PC.volume)
        b, _, _ = force(CCFMaterial(), u)
        k = 7
        h = 1e-9
        e = zeros(PC.n_points); e[k] = h
        @test -(W(u .+ e) - W(u .- e)) / (2h) / PC.volume[k] ≈ b[k] rtol = 1e-5
    end

    @testset "BAQP: keine Nullenergiemode, b = -∂W/∂u / V" begin
        mat = BAQPMaterial()
        z = [1e-6 * (-1)^i for i in 1:PC.n_points]
        b, _, _ = force(mat, z)
        @test minimum(abs, b[INTERIOR]) > 0
        # W = Σ_i V_i Σ_j ½(ω/w_i + ω/w_j) ½E ε_ij² V_j, ε_ij = Δu/ξ (linear)
        system = BBVV1D.BondSystem(PC, mat, PARAMS)
        w = BBVV1D.Storage(system, mat, PARAMS).kernel_integral
        function W(u)
            s = 0.0
            for i in BBVV1D.each_point_idx(system), bid in BBVV1D.each_bond_idx(system, i)
                j = system.bonds[bid].neighbor
                ε = (u[j] - u[i]) / (system.position[j] - system.position[i])
                s += 0.25 * (1 / w[i] + 1 / w[j]) * PARAMS.E * ε^2 * ΔX * ΔX
            end
            return s
        end
        u = 1e-9 .* randn(PC.n_points)
        b, _, _ = force(mat, u)
        k, h = 5, 1e-12
        e = zeros(PC.n_points); e[k] = h
        @test -(W(u .+ e) - W(u .- e)) / (2h) / ΔX ≈ b[k] rtol = 1e-4
    end

    @testset "momentum balance: $name" for (name, mat) in MATS
        u = 1e-6 .* randn(PC.n_points)
        b, _, _ = force(mat, u)
        if mat isa CCMaterial
            # CC: Σ b V = 2E cc Σ q_i barepsilon_i V_i ≠ 0, weil q_i am Rand abweicht
            @test_broken abs(sum(b .* PC.volume)) < 1e-8 * maximum(abs, b)
        else
            @test abs(sum(b .* PC.volume)) < 1e-8 * maximum(abs, b)
        end
    end

    @testset "Balken" begin
        EI, ρA, l, N = 2.0, 0.5, 1.0, 100
        δ = 3.015 * l / N
        A = MittlereKruemmung()
        ARK = MittlereKruemmung(; randkorrektur=true)
        B = BindungsKruemmung()
        c = sqrt(EI / ρA)
        @testset "Starrkörper und Symmetrie: $(nameof(typeof(m))) $i" for (i, m) in
                                                                         enumerate((ARK, B))
            s = BeamSystem(l, N, δ, EI, ρA, m; links=:frei, rechts=:frei)
            x = s.x[s.dofs .+ 1]
            @test maximum(abs, s.H * ones(length(x))) < 1e-8 * maximum(abs, s.H)
            @test maximum(abs, s.H * x) < 1e-8 * maximum(abs, s.H)
            @test s.H ≈ s.H'
            ω, _ = eigenfrequenzen(s)
            @test count(<(1e-3 * ω[3]), ω) == 2      # genau zwei Starrkörpermoden
        end
        @testset "ohne Randkorrektur: Drehung erzeugt Energie" begin
            s = BeamSystem(l, N, δ, EI, ρA, A; links=:frei, rechts=:frei)
            x = s.x[s.dofs .+ 1]
            @test formaenderungsarbeit(s, x) > 1e-3 * EI
        end
        @testset "Kalibrierung κ = w'' (A nur im Inneren)" for m in (A, ARK, B)
            s = BeamSystem(l, N, δ, EI, ρA, m; links=:frei, rechts=:frei)
            x = s.x[s.dofs .+ 1]
            κ = kruemmung(s, x .^ 2 ./ 2)
            bereich = m === A ? (5:N - 4) : eachindex(κ)
            @test all(isapprox.(κ[bereich], 1; rtol=1e-10))
        end
        @testset "gelenkig: Eigenfrequenzen" for m in (A, ARK, B)
            s = BeamSystem(l, N, δ, EI, ρA, m; links=:gelenkig, rechts=:gelenkig)
            ω, _ = eigenfrequenzen(s)
            @test all(isapprox.(ω[1:3], [(k * π / l)^2 * c for k in 1:3]; rtol=5e-3))
        end
        @testset "Kragarm: Endlast und Eigenfrequenz" for m in (ARK, B)
            s = BeamSystem(l, N, δ, EI, ρA, m)
            f = zeros(length(s.dofs)); f[end] = 1.0
            @test knotenwerte(s, statik(s, f))[end] ≈ l^3 / (3EI) rtol = 1e-3
            ω, _ = eigenfrequenzen(s)
            @test ω[1] ≈ (1.8751040687 / l)^2 * c rtol = 1e-3
        end
        @testset "Dynamik: Energieerhaltung" begin
            s = BeamSystem(l, N, δ, EI, ρA, B)
            x = s.x[s.dofs .+ 1]
            h = balken_dynamik(s; w0=0.01 .* x .^ 2, tend=0.05, n_out=10)
            e(k) = formaenderungsarbeit(s, h.displacement[s.dofs .+ 1, k]) +
                   0.5 * sum(s.M .* h.velocity[s.dofs .+ 1, k] .^ 2)
            @test e(size(h.displacement, 2)) ≈ e(1) rtol = 1e-3
        end
    end

    @testset "Elastica" begin
        L, N, EI, EA = 0.254, 50, 1.258e-3, 7.3e3
        sys = ElasticaSystem(L, N, 3.015 * L / N, EI, EA)
        q = elastica_gerade(sys) .+ 1e-3 .* sin.(1:2(N - 1))
        Y = elastica_lagen(sys, q, 0.0)
        @testset "Gradient" begin
            G = similar(Y)
            BBVV1D.elastica_energie!(G, sys, Y)
            k, h = 30, 1e-8
            Yp = copy(Y); Yp[2, k] += h
            Ym = copy(Y); Ym[2, k] -= h
            fd = (elastica_energie(sys, Yp) - elastica_energie(sys, Ym)) / 2h
            @test G[2, k] ≈ fd rtol = 1e-5
        end
        @testset "Dreh- und Translationsinvarianz" begin
            R = [cos(0.7) -sin(0.7); sin(0.7) cos(0.7)]
            @test elastica_energie(sys, R * Y .+ [0.1, -0.3]) ≈ elastica_energie(sys, Y) rtol = 1e-12
        end
        @testset "kleine Verformungen = lineares BP-Modell" begin
            lin = BeamSystem(L, N, 3.015 * L / N, EI, 1.0, BindungsKruemmung();
                             links=:gelenkig, rechts=:gelenkig)
            w = 1e-7 .* sin.(π .* (1:N-1) ./ N)
            q0 = elastica_gerade(sys); q0[2:2:end] .= w
            @test elastica_energie(sys, elastica_lagen(sys, q0, 0.0)) ≈
                  formaenderungsarbeit(lin, w) rtol = 1e-6
        end
        @testset "Nachknicken: Kraft bei Δ = L/2" begin
            # exakt (Elastica, Euler-Fall II): Sehne c/L = 1/2 -> p = 0.68118, F/F_cr = 1.34945
            s = ElasticaSystem(L, 100, 3.015 * L / 100, EI, EA)
            p = elastica_pfad(s, range(0.01, 0.5; length=30) .* L)
            @test p[end].F / (π^2 * EI / L^2) ≈ 1.34945 rtol = 5e-3
        end
        @testset "Dynamik: Energieerhaltung mit Sprunglast" begin
            s = ElasticaSystem(L, 40, 3.015 * L / 40, EI, EA)
            q0 = elastica_pfad(s, range(0.002, 0.05; length=6) .* L)[end].q
            ρA = 1500 * π * 1.66e-3^2 / 4
            Δt = 0.5 * 2 / elastica_omega_max(s, q0, 0.05L, ρA)
            h = elastica_dynamik(s, 0.05L, ρA; q0, Q=t -> -0.3, tend=0.01, Δt, n_out=20)
            Eg = h.W .+ h.Ekin .- h.Wext
            @test maximum(abs, Eg .- Eg[1]) < 1e-3 * maximum(h.W)
        end
    end

    @testset "Elastica-Kragarm" begin
        L, N, EI, EA = 0.254, 100, 1.258e-3, 7.3e3
        sys = ElasticaSystem(L, N, 3.015 * L / N, EI, EA; rechts_frei=true)
        κ0 = 10.0
        q0 = kragarm_bogen(sys, κ0)
        Y = kragarm_lagen(sys, q0)
        κ = elastica_kruemmung(sys, Y)
        @test all(isapprox.(κ[1:N-2], κ0; rtol=1e-10))     # Kreisbogen, auch an der Einspannung
        # Gradient gegen Differenzenquotient
        g = kragarm_gradient(sys, q0)
        k, h = 2N - 10, 1e-8
        e = zeros(2N); e[k] = h
        W(q) = elastica_energie(sys, kragarm_lagen(sys, q))
        @test g[k] ≈ (W(q0 .+ e) - W(q0 .- e)) / 2h rtol = 1e-5
        # Energie nach dem Loslassen erhalten
        ρA = 1500 * π * 1.66e-3^2 / 4
        r = kragarm_dynamik(sys, ρA; q0, tend=2e-4, Δt=2e-7, n_out=10)
        Eg = r.W .+ r.Ekin
        @test maximum(abs, Eg .- Eg[1]) < 1e-3 * Eg[1]
    end

    @testset "simulation history" begin
        bcs = [VelocityBC(t -> 1.0, [1])]
        h = redirect_stdout(devnull) do
            simulation(PC, PARAMS, OSBMaterial(), bcs; n_timesteps=100, export_freq=10,
                       vtk=false, history=true)
        end
        @test size(h.displacement) == (PC.n_points, 11)
        @test h.time[end] > 0
        @test all(isfinite, h.displacement)
    end
end
