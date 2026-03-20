# Video Script — CS4168 Spotify Tracks Analysis

**Target duration:** ~5:20 (buffer under 6:00 limit)

---

## [0:00–0:20] TITLE + HOOK — `s01_title`

> What makes a Spotify track popular? Is it the beat? The energy? The genre? We analysed two thousand tracks across five genres to find out — and the answer challenged every assumption we had.

---

## [0:20–1:00] DATASET OVERVIEW — `s02_dataset_stats`, `s03_genre_bars`

> Our dataset contains 2000 Spotify tracks spanning pop, hip-hop, indie-pop, synth-pop, and R&B. Each track has 14 audio features — danceability, energy, tempo, acousticness, and more. After cleaning 40 incomplete records, we had 1960 tracks ready for analysis.
>
> But the first surprise came early: 422 tracks — over 20% — had zero popularity. And the genres weren't evenly distributed: pop and indie-pop dominated with around 490 tracks each, while hip-hop and R&B had roughly 290.

---

## [1:00–1:40] THE KEY DISCOVERY — `s04_correlation_grid`, `s05_popularity_flat`

> Here's where things got interesting. We expected audio features to correlate with popularity. But the correlation matrix revealed something striking: popularity has near-zero correlation with every audio feature. The strongest was valence at just negative 0.10.
>
> No individual feature can linearly predict how popular a track will be. The only strong correlations were between features themselves — energy and loudness at 0.63. This finding shaped everything that followed.

---

## [1:40–2:40] CLUSTERING — `s06_silhouette_bars`, `s07_pca_scatter`, `s08_kmeans_genre`

> For clustering, we used K-Means and DBSCAN on the 14 audio features — excluding genre, since clustering should discover structure, not be told about it.
>
> The silhouette scores told a humbling story: they ranged from 0.11 to 0.16 — essentially flat. No value of k produced well-separated clusters.
>
> Still, K-Means with k=5 revealed partial structure. Hip-hop concentrated in one cluster — 144 out of 292 tracks — driven by its distinctive high speechiness. Synth-pop also grouped together. But pop and indie-pop scattered everywhere.
>
> DBSCAN struggled more. At low epsilon, over 85% of points were noise. At epsilon 3.0, it found structure comparable to K-Means — silhouette 0.19 — but with very loose neighbourhoods. Both algorithms confirmed: audio features provide a weak but detectable genre signal.

---

## [2:40–3:30] CLASSIFICATION — `s09_classification_bars`, `s10_confusion_3d`, `s11_feat_imp_clf`

> Next, we predicted whether a track is above or below the median popularity of 45. The classes were balanced — 994 low versus 966 high.
>
> Logistic Regression achieved just 60.7% — barely useful. But Random Forest jumped to 70.2%. That 10-percentage-point gap confirmed what EDA predicted: the signal is non-linear.
>
> On the test set, Random Forest reached 73.2% with balanced precision and recall. No single feature dominated — all audio features contributed roughly equally at 6 to 10 percent importance.

---

## [3:30–4:20] REGRESSION — `s12_regression_bars`, `s13_ridge_disaster`

> Predicting exact popularity scores proved far harder. Ridge Regression achieved an R-squared of just 0.018. Essentially zero. Its predictions were off by 29 points on a 100-point scale.
>
> Random Forest improved to R-squared 0.267 — fifteen times better — with an RMSE of 25. But even the best model leaves 73% of popularity variance unexplained.

---

## [4:20–5:00] KEY DECISIONS — `s14_decisions`

> One decision that improved performance: switching from Logistic Regression to Random Forest. This single change boosted classification accuracy from 60.7% to 70.2% and regression R-squared from 0.018 to 0.267 — proving the boundary is non-linear.
>
> One decision that reduced performance: Ridge Regression for popularity prediction. With R-squared of 0.018, it performed no better than predicting the mean. But this was our most informative result — it proved definitively that no linear combination of audio features predicts popularity.

---

## [5:00–5:20] CONCLUSION — `s15_conclusion`

> Audio features contain no linear signal for popularity. All predictive power comes from non-linear interactions captured by tree-based models. Even then, we explain only 27% of the variance. The rest is driven by what data cannot capture — artist fame, marketing, and cultural trends. Music popularity is fundamentally a human phenomenon that audio analysis alone cannot decode.
