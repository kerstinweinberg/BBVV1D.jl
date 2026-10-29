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
