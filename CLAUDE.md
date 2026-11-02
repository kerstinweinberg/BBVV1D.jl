# BBVV1D.jl – Hinweise für die Arbeit mit Claude Code

## Zusammenarbeit
- Kommunikation auf Deutsch, Code-Kommentare und Notizen ebenfalls deutsch.
- Auswertung und Diagramme direkt in Julia (CairoMakie/GLMakie, `scripts/vergleich/`), nicht in Paraview.
  Partikelbilder als schattierte Kugeln mit `scripts/vergleich/kugeln.jl`.
- Materialdaten realistisch in SI-Einheiten.
- Nach Änderungen an LaTeX-Texten immer das übersetzte PDF bereitstellen.
- Commit und Push nur auf ausdrückliche Bitte; neue Arbeit auf einem eigenen Branch von `main`.
- Gegenüber Gutachtern kritisch bleiben: nicht jeder Kommentar ist berechtigt.

## Modelle (src/)
- BB, OSB, CC-Varianten und BAQP für den Stab; Balken (CC, CC + RK, BP); Elastica BP und CC.
- Referenzimplementierung in 3D: https://github.com/kaipartmann/Peridynamics.jl
  (BB: bc = 18K/(πδ⁴); OSB mit linearem Kernel δ/L; Δt = 0.7·min √(2ρ/Σ V bc/L)).
- **CC ist ein eigenes Modell** (Energieäquivalenz, in 1D Mittelung über ε²), nicht die
  continuum-kinematics-inspirierte PD. Die Gewichtung mit |ξ| in `CCMaterial` ist Absicht.
  Die vorzeichenbehaftete Form steht als `CCFMaterial` (im Vergleich „CC“, die |ξ|-Variante „CC |ξ|“).
  Die Kinematik von CC nicht ändern, ohne vorher zu fragen.
- CCF hat diskret ⌊δ/Δx⌋ Nullenergiemoden; BAQP reduziert sich in 1D auf BB mit α = 2, normiert mit 1/w.
- Elastica: BP = Winkelkrümmung aus Bindungspaaren + BB-Dehnung; CC = ȳ′, ȳ″ aus RK mit Basis (ξ, ξ²),
  Kernel ξ⁻⁴, Dehnung bindungsbezogen (BAQP). Exzentrisch gedrückter Stab: `exzentrisch_pfad`.

## Stand (2026-11-02)
- Alles in `main` (PR #2). Tests: 73 bestanden, 2 als broken markiert.
- `results/` steht in `.gitignore`: Bilder und Filme mit den Skripten neu erzeugen, z. B.
  `vergleich_stab.jl`, `vergleich_balken.jl`, `vergleich_exzentrisch.jl`, `spaghetti_pd.jl`,
  `bilder_elastica.jl`, `bilder_spaghetti.jl`, `filme_vortrag.jl`
  (`julia --project=scripts/vergleich scripts/vergleich/<skript>.jl`).
- Spaghetti-Paper (Weinberg, Breaking spaghetti, Meccanica, eingereicht 2026): Revision fertig.
  Im Paper nur die BP-Elastica; die BAQP-Elastica ggf. später mit Kollegen veröffentlichen.
- Vortrag ISCM56 (5.11.2026) mit Folien und Filmen zum Spaghetti-Paper fertig.

## Offene Punkte
- Kap. 66 der Notizen: Abschnitt Spaghetti/Loslassen mit der neuen BP-Rechnung aktualisieren.
- Randkorrigierte CC-Variante (Impulserhaltung) und die 1D-Herleitung von CC.
- OSB mit α = 10E ist mit dem Zeitschritt aus bc instabil.

## LaTeX-Notizen (außerhalb des Repos)
- Kap65_Peridynamik (allgemein, endet mit „Diskretisierung und Reproducing-Kernel-Approximation“),
  Kap66_1DPeridynamik (Stab, Balken, Elastica; Hauptdatei Notizen_Peridyn1D.tex, ausführliche Fassung
  Notizen_Peridyn1DStudie.tex). Verweise zwischen den Kapiteln über das Paket xr (Präfix `K66-`),
  deshalb beide Kapitel abwechselnd übersetzen.
- Die Dateien sind Latin-1 mit CRLF: kein `sed -i` (entfernt CR) und kein `sed` mit Backslashes;
  besser auf einer UTF-8-Kopie editieren und zurückwandeln. Das Paper-Manuskript ist UTF-8.
