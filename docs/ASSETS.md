# Visual assets and provenance

The README combines conceptual cover artwork with reproducible analytical figures. The cover is not a visualization of actual transactions.

| Asset | Source | Intended use |
| --- | --- | --- |
| [follow-the-money-cover.png](assets/follow-the-money-cover.png) | Built-in image generation tool | README banner or portfolio cover |
| [model-comparison.png](assets/model-comparison.png) | Matplotlib; saved metrics | Compare ranking, alert precision and value coverage |
| [precision-recall.png](assets/precision-recall.png) | Matplotlib; holdout labels and scores | Show empirical precision-recall tradeoffs |
| [training-activity.png](assets/training-activity.png) | Matplotlib; training-period SQL EDA | Explain the generator's activity and planted fraud |
| [time-safety.png](assets/time-safety.png) | Matplotlib; split metadata and feature rule | Explain chronological evaluation and feature availability |

## Social post cards

These square cards are ready for a LinkedIn carousel or project announcement. They are conceptual artwork generated with the built-in image-generation tool; they do not add empirical evidence to the model results.

| Asset | Suggested post use |
| --- | --- |
| [follow-the-money-post-cover.png](assets/social/follow-the-money-post-cover.png) | Opening card: project title and mobile-wallet network theme |
| [follow-the-money-workflow-card.png](assets/social/follow-the-money-workflow-card.png) | Second card: generate/load -> DuckDB/EDA -> features -> models -> holdout |
| [follow-the-money-results-card.png](assets/social/follow-the-money-results-card.png) | Third card: explain why historical context and value coverage create a tradeoff |

The four analytical figures use navy, teal, slate, and amber. Charts retain axis scales, definitions, data scope, and simulation labels. Alternative text accompanies each image in the README.

## Regenerate analytical figures

From the project root, after a successful pipeline run:

```powershell
.venv/Scripts/python.exe scripts/visualize.py --results outputs --out docs/assets --snapshot-dir docs/results
```

The script checks that the run and artifact audit passed, recalculates metrics from saved scores, and writes four PNGs plus compact JSON snapshots. Input hashes are recorded in [run-summary.json](results/run-summary.json). Score rows remain in local, Git-ignored outputs. The timeline's feature-history panel is schematic, not to scale.

## Cover generation

Mode: **built-in image generation**, not the CLI/API fallback. No external image download or provider logo was used. The generated image was copied into the project as `docs/assets/follow-the-money-cover.png`.

Final generation prompt:

```text
Use case: stylized-concept
Asset type: landscape GitHub README cover for a portfolio machine-learning project named Follow the Money.
Primary request: an elegant, professionally art-directed editorial illustration about mobile-wallet transaction networks and fraud detection. Show a single clean modern smartphone standing at a slight isometric angle, its screen displaying a minimalist wallet icon. Surround it with a carefully arranged network of small nodes and fine connections; most connections are muted blue and teal, while a small concentrated incoming-transfer cluster and onward path glow warm amber, suggesting an anomaly discovered through historical context. Include subtle stacked database cylinders and a few translucent rectangular data cards, with no fake numbers or charts. The composition should feel like an analytical research publication, restrained and precise, not a cybercrime movie.
Style: polished dimensional illustration with matte surfaces, fine lines, subtle grain and generous negative space. Wide panoramic 2.4:1 composition suitable for a README banner. Dark midnight navy background, cool teal and off-white highlights, a restrained amber anomaly accent. Subject on right half; left half clear for the title.
Text (verbatim): "FOLLOW THE MONEY" as large crisp off-white editorial typography on the left, with smaller subtitle "Mobile-wallet fraud analysis" and small understated label "SIMULATED DATA · REPRODUCIBLE ML".
Constraints: text exactly as supplied. No company logos, no provider branding, no shields, padlocks, hooded hackers, robots, brains, coins with currency symbols, performance numbers, axes, charts, or claims of real-world accuracy. The network is conceptual artwork, not an empirical transaction graph. Beautiful and legible at GitHub content width.
```

## Sharing

The cover is conceptual artwork; identify it as such when shared separately. Keep the simulation labels and metric footnotes visible when sharing charts. These results support a reproducible portfolio demonstration, not claims of provider performance or prevented losses.
