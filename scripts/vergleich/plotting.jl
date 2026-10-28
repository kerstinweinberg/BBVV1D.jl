# Modellunabhängige Vergleichsdiagramme (CairoMakie).
# Ein "run" ist ein NamedTuple (; label, hist, volume, rho) mit hist aus
# simulation(...; history=true). Die Referenz ist eine Funktion ref(x, t) -> u.
# Damit lassen sich später auch Balkenmodelle mit denselben Funktionen vergleichen.

using CairoMakie

# feste Reihenfolge der Kategorien (validierte Standardpalette, Slots 1-4)
const FARBEN = Dict("BB" => "#2a78d6", "CC" => "#eb6834", "OSB" => "#1baf7a",
                    "BB-3D" => "#eda100", "OSB-3D" => "#e87ba4")
const FARBEN_REST = ["#eda100", "#e87ba4", "#008300"]
const REF_FARBE = "#3d3d3a"
const REF_LABEL = "analytisch"

farbe(label, i=1) = get(FARBEN, label, FARBEN_REST[mod1(i, length(FARBEN_REST))])

# frühere Kurven breiter, damit nahezu deckungsgleiche Modelle sichtbar bleiben
linienbreite(i, n) = 2.0 + 1.2 * (n - i) / max(n - 1, 1)

# lesbare Achsenwerte für log-Skalen
const LOG_TICKS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
log_ticks(vmin, vmax) = filter(v -> vmin / 1.5 ≤ v ≤ vmax * 1.5, LOG_TICKS)

const THEMA = merge(theme_latexfonts(),
                    Theme(fontsize=12, linewidth=2,
                          Axis=(xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08),
                                topspinevisible=false, rightspinevisible=false),
                          Legend=(framevisible=false,)))

function speichern(fig, pfad)
    mkpath(dirname(pfad))
    save(pfad * ".pdf", fig)
    save(pfad * ".png", fig; px_per_unit=2)
    return pfad
end

zeitindex(hist, t) = argmin(abs.(hist.time .- t))

# --- Auswertungsgrößen ---------------------------------------------------------

"relativer L2-Fehler e(t) = ‖u - u_ref‖ / max_t ‖u_ref‖"
function fehler(run, ref)
    x = run.hist.position
    uref = [ref(xi, t) for xi in x, t in run.hist.time]
    diff = sqrt.(vec(sum(abs2, run.hist.displacement .- uref; dims=1)))
    return diff ./ maximum(sqrt.(vec(sum(abs2, uref; dims=1))))
end

"kinetische, innere und Gesamtenergie pro Querschnittsfläche; W_int = -½ Σ b u V (linear)"
function energie(run)
    h = run.hist
    V = run.volume
    Ekin = vec(0.5 * run.rho * sum(h.velocity .^ 2 .* V; dims=1))
    Eint = vec(-0.5 * sum(h.b_int .* h.displacement .* V; dims=1))
    return (; Ekin, Eint, Eges=Ekin .+ Eint)
end

# --- Diagramme ------------------------------------------------------------------

"Dispersion: curves = [(label, kΔx, ω/(ck)), ...] oder mit Farbe als 4. Eintrag"
function plot_dispersion(curves; title="Dispersion")
    with_theme(THEMA) do
        fig = Figure(size=(620, 340))
        ax = Axis(fig[1, 1]; xlabel=L"k\,\Delta x", ylabel=L"\omega / (c\,k)", title,
                  xticks=([0, π / 4, π / 2, 3π / 4, π],
                          [L"0", L"\pi/4", L"\pi/2", L"3\pi/4", L"\pi"]))
        hlines!(ax, 1.0; color=REF_FARBE, linestyle=:dash, linewidth=1.5, label="lokal")
        for (i, c) in enumerate(curves)
            label, k, w = c
            lines!(ax, k, w; color=length(c) ≥ 4 ? c[4] : farbe(label, i),
                   linestyle=length(c) ≥ 5 ? c[5] : :solid, label=label)
        end
        xlims!(ax, 0, π)
        ylims!(ax, 0, nothing)
        Legend(fig[1, 2], ax)
        fig
    end
end

"Momentaufnahmen u(x) zu den Zeiten ts, alle Modelle + Referenz"
function plot_snapshots(runs, ts; ref=nothing, tnorm=1.0, tlabel="t")
    with_theme(THEMA) do
        fig = Figure(size=(620, 170 * length(ts) + 60))
        axs = Axis[]
        for (row, t) in enumerate(ts)
            ax = Axis(fig[row, 1]; ylabel=L"u\;[\mathrm{m}]",
                      title=string(tlabel, " = ", round(t / tnorm; digits=2)),
                      titlealign=:left, titlesize=11)
            push!(axs, ax)
            x = runs[1].hist.position
            if ref !== nothing
                lines!(ax, x, ref.(x, t); color=REF_FARBE, linestyle=:dash, linewidth=1.5,
                       label=REF_LABEL)
            end
            for (i, r) in enumerate(runs)
                it = zeitindex(r.hist, t)
                lines!(ax, r.hist.position, r.hist.displacement[:, it];
                       color=farbe(r.label, i), linewidth=linienbreite(i, length(runs)),
                       label=r.label)
            end
            row < length(ts) && hidexdecorations!(ax; grid=false)
        end
        axs[end].xlabel = L"x\;[\mathrm{m}]"
        linkaxes!(axs...)
        Legend(fig[0, 1], axs[1]; orientation=:horizontal)
        fig
    end
end

"x-t-Diagramm u(x,t) je Modell (Kleinmultiples, gemeinsame divergente Skala)"
function plot_xt(runs; ref=nothing, tnorm=1.0, tlabel=L"t")
    with_theme(THEMA) do
        felder = Tuple{String,Vector{Float64},Vector{Float64},Matrix{Float64}}[]
        if ref !== nothing
            h = runs[1].hist
            push!(felder, (REF_LABEL, h.position, h.time,
                           [ref(xi, t) for xi in h.position, t in h.time]))
        end
        for r in runs
            push!(felder, (r.label, r.hist.position, r.hist.time, r.hist.displacement))
        end
        umax = maximum(maximum(abs, f[4]) for f in felder)
        fig = Figure(size=(260 * length(felder) + 80, 360))
        hm = nothing
        for (col, (label, x, t, u)) in enumerate(felder)
            ax = Axis(fig[1, col]; title=label, xlabel=L"x\;[\mathrm{m}]",
                      ylabel=col == 1 ? tlabel : "")
            hm = heatmap!(ax, x, t ./ tnorm, u; colormap=Reverse(:RdBu),
                          colorrange=(-umax, umax))
            col > 1 && hideydecorations!(ax; grid=false)
        end
        Colorbar(fig[1, length(felder) + 1], hm; label=L"u\;[\mathrm{m}]")
        fig
    end
end

"Zeitverlauf u(x_p, t) an Messpunkten xps"
function plot_probe(runs, xps; ref=nothing, tnorm=1.0, tlabel=L"t")
    with_theme(THEMA) do
        fig = Figure(size=(620, 190 * length(xps) + 60))
        axs = Axis[]
        for (row, xp) in enumerate(xps)
            ax = Axis(fig[row, 1]; ylabel=L"u\;[\mathrm{m}]",
                      title=string("x = ", round(xp; digits=3), " m"),
                      titlealign=:left, titlesize=11)
            push!(axs, ax)
            t = runs[1].hist.time
            if ref !== nothing
                lines!(ax, t ./ tnorm, ref.(xp, t); color=REF_FARBE, linestyle=:dash,
                       linewidth=1.5, label=REF_LABEL)
            end
            for (i, r) in enumerate(runs)
                ip = argmin(abs.(r.hist.position .- xp))
                lines!(ax, r.hist.time ./ tnorm, r.hist.displacement[ip, :];
                       color=farbe(r.label, i), linewidth=linienbreite(i, length(runs)),
                       label=r.label)
            end
            row < length(xps) && hidexdecorations!(ax; grid=false)
        end
        axs[end].xlabel = tlabel
        linkxaxes!(axs...)
        Legend(fig[0, 1], axs[1]; orientation=:horizontal)
        fig
    end
end

"relativer Fehler gegenüber der Referenz über der Zeit"
function plot_error(runs, ref; tnorm=1.0, tlabel=L"t")
    with_theme(THEMA) do
        fig = Figure(size=(520, 320))
        ax = Axis(fig[1, 1]; xlabel=tlabel, ylabel=L"\Vert u-u_\mathrm{ref}\Vert / \max\Vert u_\mathrm{ref}\Vert")
        for (i, r) in enumerate(runs)
            lines!(ax, r.hist.time ./ tnorm, fehler(r, ref); color=farbe(r.label, i),
                   linewidth=linienbreite(i, length(runs)), label=r.label)
        end
        ylims!(ax, 0, nothing)
        axislegend(ax; position=:lt)
        fig
    end
end

"Energiebilanz je Modell (Kleinmultiples: kinetisch, innere, gesamt)"
function plot_energy(runs; tnorm=1.0, tlabel=L"t")
    with_theme(THEMA) do
        fig = Figure(size=(260 * length(runs) + 40, 320))
        axs = Axis[]
        for (col, r) in enumerate(runs)
            en = energie(r)
            t = r.hist.time ./ tnorm
            c = farbe(r.label, col)
            ax = Axis(fig[1, col]; title=r.label, xlabel=tlabel,
                      ylabel=col == 1 ? L"E/A\;[\mathrm{J/m^2}]" : "")
            push!(axs, ax)
            lines!(ax, t, en.Eges; color=REF_FARBE)
            lines!(ax, t, en.Ekin; color=c, linestyle=:dash, linewidth=1.5)
            lines!(ax, t, en.Eint; color=c, linestyle=:dot, linewidth=1.5)
            col > 1 && hideydecorations!(ax; grid=false)
        end
        linkaxes!(axs...)
        # Legende nur für die Linienart, neutral eingefärbt (Farbe = Modell, steht im Titel)
        elemente = [LineElement(color=REF_FARBE),
                    LineElement(color=REF_FARBE, linestyle=:dash, linewidth=1.5),
                    LineElement(color=REF_FARBE, linestyle=:dot, linewidth=1.5)]
        Legend(fig[0, :], elemente, ["gesamt", "kinetisch", "innere"];
               orientation=:horizontal)
        fig
    end
end

"Parameterstudie: daten = [(label, xs, fehler[, farbe[, linienart]]), ...], doppelt-logarithmisch"
function plot_study(daten; xlabel, ylabel=L"\max_t\, e(t)", xscale=log10, yscale=log10,
                    title="", xticks=Makie.automatic, legendpos=:lt)
    with_theme(THEMA) do
        fig = Figure(size=(520, 340))
        es_alle = reduce(vcat, [d[3] for d in daten])
        yticks = yscale === log10 ? log_ticks(extrema(es_alle)...) : Makie.automatic
        ax = Axis(fig[1, 1]; xlabel, ylabel, xscale, yscale, title, xticks, yticks)
        for (i, d) in enumerate(daten)
            label, xs, es = d
            scatterlines!(ax, xs, es; color=length(d) ≥ 4 ? d[4] : farbe(label, i),
                          linestyle=length(d) ≥ 5 ? d[5] : :solid, markersize=9, label=label)
        end
        axislegend(ax; position=legendpos)
        fig
    end
end
