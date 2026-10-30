# Partikelbilder: die diskretisierte Struktur als schattierte Kugeln (wie in den
# ParaView-Bildern des GAMM-Artikels), mit Makie statt ParaView. Backend-unabhängig:
# GLMakie für Filme und große Punktmengen, CairoMakie für Standbilder.

# Vorher ein Backend laden (using GLMakie oder using CairoMakie), das Makie bereitstellt.
using Printf

const KUGEL_THEMA = merge(theme_latexfonts(), Theme(fontsize=14))
const KUGEL_KARTE = :coolwarm        # entspricht "Cool to Warm" in ParaView

"""
    kugelachse(fig_pos; draufsicht=true, title="")

Achse ohne Rahmen mit gleichen Maßstäben und Parallelprojektion. `draufsicht = true` schaut
von oben auf die x-z-Ebene der ebenen Modelle (dort als x-y gezeichnet).
"""
function kugelachse(pos; draufsicht::Bool=true, title="", azimuth=-0.35π, elevation=0.12π)
    # ebene Modelle: 2D-Achse mit gleichem Maßstab (eine Axis3 säße in einem Würfel, in dem ein
    # flacher Stab nur einen schmalen Streifen füllt); die Kugeln bleiben schattiert
    ax = draufsicht ? Axis(pos; aspect=DataAspect(), title) :
         Axis3(pos; aspect=:data, perspectiveness=0.0, title, azimuth, elevation)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

"Achsgrenzen g = (xmin, xmax, ymin, ymax) für 2D- und 3D-Kugelachsen"
grenzen!(ax::Axis, g, r) = limits!(ax, g[1], g[2], g[3], g[4])
grenzen!(ax::Axis3, g, r) = limits!(ax, g[1], g[2], g[3], g[4], -r, r)

"Kugeln an den Punkten `p` (Vector{Point3f}) mit Farbe `feld`"
kugeln!(ax, p, feld; radius, colorrange, colormap=KUGEL_KARTE) =
    meshscatter!(ax, p; markersize=radius, color=feld, colormap, colorrange,
                 shading=Makie.automatic)

"reale Knoten 0..N einer ElasticaSystem-Lagematrix als Punkte in der Ebene z = 0"
elastica_punkte(sys, Y) = [Point3f(Y[1, c], Y[2, c], 0) for c in (sys.nmax + 1):(sys.nmax + 1 + sys.N)]

"Grenzen [xmin, xmax, ymin, ymax] über alle Lagematrizen, mit Rand"
function grenzen(sys, Ys; rand=0.05)
    p = reduce(vcat, elastica_punkte.(Ref(sys), Ys))
    x = getindex.(p, 1); y = getindex.(p, 2)
    dx, dy = maximum(x) - minimum(x), maximum(y) - minimum(y)
    r = rand * max(dx, dy)
    return (minimum(x) - r, maximum(x) + r, minimum(y) - r, maximum(y) + r)
end

"""
    elastica_film(datei, sys, Ys, felder, texte; radius, colorrange, colormap, label, fps=30)

Film aus den Lagematrizen `Ys` mit den Feldern `felder` (je Bild ein Vektor über die Knoten)
und einem Text je Bild. Feste Achsgrenzen über alle Bilder.
"""
function elastica_film(datei, sys, Ys, felder, texte; radius, colorrange,
                       colormap=KUGEL_KARTE, label="", fps::Int=30, size=(900, 600))
    g = grenzen(sys, Ys)
    pkt = Observable(elastica_punkte(sys, Ys[1]))
    fld = Observable(felder[1])
    txt = Observable(texte[1])
    fig = with_theme(KUGEL_THEMA) do
        f = Figure(; size)
        ax = kugelachse(f[1, 1]; title=txt)
        ms = kugeln!(ax, pkt, fld; radius, colorrange, colormap)
        grenzen!(ax, g, radius)
        Colorbar(f[1, 2], ms; label, height=Relative(0.6))
        f
    end
    mkpath(dirname(datei))
    record(fig, datei, eachindex(Ys); framerate=fps) do k
        pkt[] = elastica_punkte(sys, Ys[k])
        fld[] = felder[k]
        txt[] = texte[k]
    end
    return datei
end
