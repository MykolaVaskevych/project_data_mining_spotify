import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    import pandas as pd
    import polars.selectors as cs

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        silhouette_score,
        classification_report,
        confusion_matrix,
        mean_squared_error,
        r2_score,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        RandomForestRegressor,
        GradientBoostingRegressor,
    )

    from scipy import stats
    import statsmodels.api as sm

    import matplotlib.pyplot as plt

    import seaborn as sns

    alt.data_transformers.enable("vegafusion")
    return (
        ColumnTransformer,
        DBSCAN,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        KMeans,
        LogisticRegression,
        OneHotEncoder,
        PCA,
        Pipeline,
        RandomForestClassifier,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        alt,
        classification_report,
        confusion_matrix,
        cross_val_score,
        mean_squared_error,
        mo,
        np,
        pd,
        pl,
        plt,
        r2_score,
        silhouette_score,
        sns,
        stats,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # CS4168 Data Mining Project - Spotify Tracks Analysis (Group 14)

    In this notebook we analyses an excerpt from the  Spotify Tracks Dataset available at:
    https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset


    We perform our analysis with the objective of extracting information from the provided data on how reliably attributes of data instances can predict their popularity features.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Exploratory Data Analysis

    We begin with an **Exploratory Data Analysis (EDA)** of our dataset. We do this to aid our understand and summarize the data that we our working with. We will accomplish this using quantitative and visual methods, following up on observations in the data as they appear ensuring not to take actions on assumptions that we may held on what relationships/structures should appear in the data.

    ### 1.1 Dataset Overview
    """)
    return


@app.cell
def _(mo, pl):
    raw_df = pl.read_csv("tracks2026.csv")
    raw_df

    eda_shape = raw_df.shape
    eda_schema = pl.DataFrame(
        {
            "Column": list(raw_df.schema.keys()),
            "Type": [str(v) for v in raw_df.schema.values()],
        }
    )

    mo.vstack([
        mo.md(f"- **Observations:** {eda_shape[0]} \n- **Features:** {eda_shape[1]}"),
        eda_schema,
        mo.md(f"**Sample data observations:**"),
        raw_df.head()
    ])
    return (raw_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    With the dataset loaded, we perform an initial inspection of its structure. We can see that the dataset is comprised of Spotify tracks across 5 genres (pop, indie-pop, synth-pop, r-n-b, hip-hop). There are 2,000 tracks (observations) and 17 columns (features), including both numerical and categorical features. Feature descriptors are taken from the datasets hugging face page to help contextualize their  meaning, scale and role within the analysis:


    | Feature       | Description |
    |--------------------|-------------|
    | track_id           | The Spotify ID for the track |
    | popularity         | A value between 0 and 100 indicating track popularity. Based on total plays and recency. Duplicate tracks are rated independently. Artist and album popularity derive from track popularity |
    | duration_ms        | The track length in milliseconds |
    | explicit           | Whether the track has explicit lyrics (`true` = yes, `false` = no or unknown) |
    | danceability       | Measure (0.0–1.0) of how suitable a track is for dancing based on tempo, rhythm, and beat stability |
    | energy             | Measure (0.0–1.0) of intensity and activity; higher values indicate faster, louder, noisier tracks |
    | key                | The musical key using pitch class notation (e.g., 0 = C, 1 = C♯/D♭). `-1` indicates no key detected |
    | loudness           | Overall loudness of a track in decibels (dB) |
    | mode               | Indicates modality: major (`1`) or minor (`0`) |
    | speechiness        | Measure (0.0–1.0) of spoken word presence. >0.66 = mostly speech, 0.33–0.66 = mixed, <0.33 = mostly music |
    | acousticness       | Confidence measure (0.0–1.0) of whether the track is acoustic |
    | instrumentalness   | Likelihood (0.0–1.0) that the track contains no vocals |
    | liveness           | Likelihood (0.0–1.0) that the track was performed live (>0.8 indicates strong probability) |
    | valence            | Measure (0.0–1.0) of musical positivity (higher = happier, lower = sadder) |
    | tempo              | Estimated tempo in beats per minute (BPM) |
    | time_signature     | Estimated time signature (range 3–7, representing 3/4 to 7/4) |
    | track_genre        | The genre of the track |

    With this foundation, we proceed to a more detailed analysis by applying independant processing steps to the numerical and categorical data respectivly.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    In this section we begin are true analysis of the data. Focusing on techniques for analysis the numerical attributes across all 2,000 observations. To do this we must first partition our numeric and cateigorical features. For some features this is obvious; others require a bit more thougth.

    | Feature          | Category                     | Reason                                                                |
    | ---------------- | ---------------------------- | --------------------------------------------------------------------- |
    | track_id         | Categorical                  | Unique identifier; no numeric meaning or ordering                     |
    | popularity       | Numerical (ordinal) | Bounded score (0–100) representing ranked popularity intensity        |
    | duration_ms      | Numerical (continuous)       | True quantitative measurement of time duration                        |
    | explicit         | Categorical (binary)         | Boolean indicator; represents a class (explicit vs not)               |
    | danceability     | Numerical (continuous)       | Real-valued perceptual score (0–1) with meaningful scale              |
    | energy           | Numerical (continuous)       | Continuous intensity measure derived from audio signal features       |
    | key              | Categorical (nominal)        | Musical pitch class; integer encoding has no linear meaning           |
    | loudness         | Numerical (continuous)       | Physical measurement in dB; meaningful magnitude and differences      |
    | mode             | Categorical (binary)         | Two discrete classes: major vs minor; encoding is not magnitude-based |
    | speechiness      | Numerical (continuous)       | Probability-like continuous measure of speech content                 |
    | acousticness     | Numerical (continuous)       | Continuous confidence score of acoustic likelihood                    |
    | instrumentalness | Numerical (continuous)       | Probability-like measure of absence of vocals                         |
    | liveness         | Numerical (continuous)       | Continuous probability of live recording presence                     |
    | valence          | Numerical (continuous)       | Continuous affective score (0–1 emotional positivity scale)           |
    | tempo            | Numerical (continuous)       | Real-valued physical quantity (BPM)                                   |
    | time_signature   | Categorical                  | Discrete structural classes (3/4–7/4); not linear or metric           |
    | track_genre      | Categorical                  | Nominal label indicating genre class                                  |

    As we examined the feature set we categorized that the track_id feature should be treated as an _irrelevant attribute_ (dropped) as it functions as an identifier rather than a predictor.
    """)
    return


@app.cell
def _(raw_df):
    """
    Droppoing irelevant features and seperating numeric features from caetgorical features in the dataset
    """

    df = raw_df.drop("track_id")

    numeric_cols = [
        "popularity", "duration_ms", "danceability", "energy",
        "loudness", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]
    numeric_df = raw_df.select(numeric_cols)

    caetgorical_cols = [
        "track_genre", "time_signature", "mode", "key", "explicit"
    ]
    categorical_df = raw_df.select(caetgorical_cols)
    return categorical_df, numeric_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 EDA of Numerical Data

    Now that we have iscolated the numeric data (described below) we can begin to inspect it's quality. To accomplish this we will inspect and handle any missing values, investigate outliers and normalize data where appropriate.
    """)
    return


@app.cell
def _(numeric_df):
    numeric_df.head()
    return


@app.cell
def _(numeric_df):
    numeric_df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.2.1 Missing Values

    We seen when we described the numeric data with Polars that there are features with missing values (null_count) withing the numeric data. Below we narrow down which features are missing any values from any of the 2,000 observations.
    """)
    return


@app.cell
def _(numeric_df, pl, raw_df):
    null_summary = (
        numeric_df.select([
            pl.col(c).null_count().alias(c)
            for c in numeric_df.columns
        ])
        .transpose(include_header=True, column_names=["null_count"])
        .with_columns([
            ((pl.col("null_count") / raw_df.height) * 100)
            .round(2)
            .alias("percent")
        ])
        .filter(pl.col("null_count") > 0)
        .sort("null_count", descending=True)
    )

    print(null_summary)
    return (null_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have established exactly which features have missing values we must investigate whether missing values are _missing at random_(MAR), _missing completly at random_(MCAR) or _missing not at random_(MNAR); before finally determining the best treatment for each feature.


    Our domain knowledge tells us that all songs have a popularity, danceability, energy, tempo and loudness. Therfore the values are likly MCAR. However we will conduct some imperical tests to help support this intuition.

    We will refer to those values with null values as _target_attribute_

    We perform Welch's t-test on _target_attributes_ to help assess whether they are missing completly at random. When null values are Missing Complety At Random the probability that a value is missing is independant of both observerd and unobserved data. Welchs t-test provides a way to evaluate this claim for a given _target_attributes_ by comparing the difference between the mean of other attributes when the _target_attribute_ is null versus populated. Conventinally 0.05 is used as the threshold for this test, where a p_value of > 0.05 does not support a strong difference in mean between observations where the _target_attribue_ is mising or holds a value.
    """)
    return


@app.cell
def _(null_summary, numeric_df, pl, stats):
    """
    mcar_check
    Computes welchs t-test to compare the difference in mean between observations of the target attribute when 
    returns 
    """
    def mcar_check(df: pl.DataFrame, col_missing_values: str):
        results = []

        # Adds an bool indicator column, 1 if a value is missing for the col_missing_values
        df_flag = df.with_columns(
            pl.col(col_missing_values).is_null().cast(pl.Int8).alias("R")
        )

        for col in df.columns:
            if col in (col_missing_values, "R"):
                continue

            # convert dataframe to pandas for scipy
            pdf = df_flag.select([col, "R"]).drop_nulls().to_pandas()

            group0 = pdf[pdf["R"] == 0][col] # rows where value is present for col
            group1 = pdf[pdf["R"] == 1][col] # rows where value is null for col

            if len(group0) > 1 and len(group1) > 1:
                stat, p = stats.ttest_ind(group0, group1, equal_var=False)
                results.append((col, "t-test", p))


        return pl.DataFrame(results, schema=["feature", "test", "p_value"])

    for attribute_with_null_values in null_summary["column"]:
        mcar_results = mcar_check(numeric_df, attribute_with_null_values)
        print("Welchs t-test for attribute with null values:", attribute_with_null_values)
        print(mcar_results.sort("p_value"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After performing Welch's t-test, we see all p_values are greater than 0.05 supporting our intuition based of our domain knowledge that the features missing values popularity, danceability, energy, tempo and loudness should be present and are MCAR. The question then becomes how should the observations with null attributes be handeled.

    We know from an earlier description that observations with missing values compose approaximatly 2% of the total observations. We also know using our own domain knowledge that popularity, danceability, energy, tempo and loudness cannot be reliably predicted using any quantitative means; therfore the decision is made to drop those observations who contain missing values.
    """)
    return


@app.cell
def _(categorical_df, mo, numeric_df, pl):
    numeric_df_indexed = numeric_df.with_row_index("row_id")
    categorical_df_indexed = categorical_df.with_row_index("row_id")

    before = numeric_df.height
    clean_numeric_df = numeric_df_indexed.drop_nulls()
    after = clean_numeric_df.height
    clean_categorical_df = categorical_df_indexed.filter(
        pl.col("row_id").is_in(clean_numeric_df["row_id"])
    )
    clean_categorical_df = categorical_df_indexed.join(
        clean_numeric_df.select("row_id"),
        on="row_id",
        how="inner"
    )

    clean_numeric_df = clean_numeric_df.drop("row_id")
    clean_categorical_df = clean_categorical_df.drop("row_id")

    mo.md(f"""
    Number of observations before dropping missing values: **{before}**

    Number of observations after dropping missing values: **{after}**
    """)
    return clean_categorical_df, clean_numeric_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.2.2 Outliers

    Now we have dealt with missing values in are numerical data we can move onto inspecting the distribution of this data. We use box plots to acomoplish this; summarizing the 5 numer summary and indicating outliers. We analye each numeric feature with box plots:
    """)
    return


@app.cell
def _(clean_numeric_df, plt):
    pandas_numeric_df = clean_numeric_df.to_pandas()
    pandas_numeric_df.plot(
        kind="box",
        subplots=True,
        layout=(4, 4),   # adjust based on number of features
        figsize=(12, 10),
        sharex=False,
        sharey=False
    )

    plt.tight_layout()
    plt.show()
    return (pandas_numeric_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What jumps out straigth away on inspecting the boxplots is the prevelance of outliers in many of the features. Generating the histograms for these features may give another perspective on what the root cause of these outliers may be.
    """)
    return


@app.cell
def _(pandas_numeric_df, plt):
    pandas_numeric_df.hist(bins=30, figsize=(12, 8))
    plt.tight_layout()
    plt.show()
    plt.close("")
    return


@app.cell
def _(clean_numeric_df, mo, pl):
    zero_popularity_count = clean_numeric_df.filter(pl.col("popularity") == 0).shape[0]
    percenteage_zero_pop_count = (zero_popularity_count / len(clean_numeric_df.rows()) ) * 100
    mo.md(f"""
    Now that we have the box plots and histograms for are numeric data we can analyse each feature and decide what the best course of action is for each feature; tailoring our data processing to it's unique charachteristics.

    The **popularity**
    exhibts no significant outliers using the IQR rule. However, it shows a pronounced spike at 0 with {zero_popularity_count}({percenteage_zero_pop_count:.2f}%) of tracks having zero popularity. This indicates a zero-inflated distribution. We hypothsis this could be due to obscure or delisted tracks, platform bias, new listings or censoring altough this cannot be validated through data alone.

    When excluding zero values, the remaining distribution appears multi-modal. The presence of a large mass at zero may introduce challenges for modelling, particularly for regression, where zero-inflation can violate standard assumptions and bias predictions. Distance-based methods such as K-NN may also be affected, as the high frequency of identical values reduces the feature’s discriminative power.

    As modelling techniques have not yet been finalised, no preprocessing transformations (e.g., zero-handling or feature engineering) will be applied at this stage to avoid premature assumptions about the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **duration_ms** feature exhibts a rigth-skew, with most tracks clustered at shorter durations and a small number of substantially longer tracks appearing as outliers in the box plot. This pattern is consistent with expectations for music data and does not inherently violate modelling assumptions.

    However, the presence of a long right tail and high-value outliers may disproportionately influence scale-sensitive models. Additionally, differences in duration are more meaningful interpreted on a relative scale (e.g., the perceived difference between 2 and 4 minutes is greater than between 1 and 2 minutes) by music listeners.

    To address this, a log transformation is applied to duration_ms, which reduces skewness, compresses extreme values, and improves the feature’s suitability for modelling while preserving ordinal relationships.
    """)
    return


@app.cell
def _(clean_numeric_df, pl, plt):
    min_val = clean_numeric_df.select(pl.col("duration_ms").min()).item()

    new_clean_numeric_df = clean_numeric_df.with_columns(
        (pl.col("duration_ms") + abs(min_val) + 1)
        .log()
        .alias("duration_ms")
    )

    duration_ms_after = new_clean_numeric_df['duration_ms'].to_pandas()
    duration_ms_after.hist(bins=30, figsize=(7, 5))

    plt.title("Log-Transformed Duration MS Distribution")
    plt.xlabel("Log(Duration (ms))")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    plt.close("all")
    return (new_clean_numeric_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **danceability**, **energy**, **acousticness**, **liveness** and **valance** features are bounded within the expected rangs [0,1], and no values fall outside this range. Observations appearing as outliers in each features respective box plot represent valid edge cases rather than extreme erroneous outliers. As a result we take no action on the features as to do so would risk distorting the semantic interpretation of the features and discarding potentially informative variation.
    """)
    return


@app.cell
def _(mo, new_clean_numeric_df, pl):

    zero_instrumentalness_count = new_clean_numeric_df.filter(pl.col("instrumentalness") == 0).shape[0]
    zero_instrumentalness_percent = (zero_instrumentalness_count / len(new_clean_numeric_df.rows())) * 100

    low_instrumentalness_count = new_clean_numeric_df.filter(pl.col("instrumentalness") < 0.2).shape[0]
    low_instrumentalness_percent = (low_instrumentalness_count / len(new_clean_numeric_df.rows())) * 100

    mo.md(f"""

    The instrumentalness feature exhibits a zero-inflated and heavily right-skewed distribution, with {zero_instrumentalness_count} observations ({zero_instrumentalness_percent:.2f}%) equal to 0 and {low_instrumentalness_count} observations ({low_instrumentalness_percent:.2f}%) falling below 0.2. This indicates that the majority of tracks in the dataset have low predicted instrumental confidence, with only a small subset showing strong instrumental characteristics.

    This pattern likely reflects the presence of vocal-heavy music and may also include speech-oriented or non-musical audio content within the dataset. However, instrumentalness is a confidence score rather than a direct measurement, meaning low values may not strictly indicate the absence of instrumentation.

    Given the strong separation between near-zero values and the small proportion of high-confidence instrumental tracks, a binary representation is considered for downstream analysis. To evaluate this, we examine the impacts of different thresholds for binarization on other features of the dataset to examine whether the 5 number summary group-wise distributions remain stable across thresholds.
    """)

    return


@app.cell
def _(new_clean_numeric_df, plt):
    pandas_df = new_clean_numeric_df.to_pandas()

    thresholds = [0.3, 0.5, 0.8]
    fig2, axes = plt.subplots(5, len(thresholds), figsize=(15, 15))

    for ax, t in zip(axes[0], thresholds):
        grouped = pandas_df.assign(group=pandas_df["instrumentalness"] > t)
        grouped.boxplot(column="energy", by="group", ax=ax)
        ax.set_title(f"threshold = {t}")

    for ax, t in zip(axes[1], thresholds):
        grouped = pandas_df.assign(group=pandas_df["instrumentalness"] > t)
        grouped.boxplot(column="danceability", by="group", ax=ax)
        ax.set_title(f"threshold = {t}")

    for ax, t in zip(axes[2], thresholds):
        grouped = pandas_df.assign(group=pandas_df["instrumentalness"] > t)
        grouped.boxplot(column="valence", by="group", ax=ax)
        ax.set_title(f"threshold = {t}")

    for ax, t in zip(axes[3], thresholds):
        grouped = pandas_df.assign(group=pandas_df["instrumentalness"] > t)
        grouped.boxplot(column="acousticness", by="group", ax=ax)
        ax.set_title(f"threshold = {t}")

    for ax, t in zip(axes[4], thresholds):
        grouped = pandas_df.assign(group=pandas_df["instrumentalness"] > t)
        grouped.boxplot(column="loudness", by="group", ax=ax)
        ax.set_title(f"threshold = {t}")

    plt.suptitle("")
    plt.tight_layout()
    plt.show()
    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A threshold of instrumentalness > 0.5 is used to classify tracks as instrumental, aligning with common interpretability thresholds for this feature while preserving meaningful signal.
    """)
    return


@app.cell
def _(clean_categorical_df, new_clean_numeric_df, pl):
    categorical_df_1 = clean_categorical_df.with_columns(
        (new_clean_numeric_df["instrumentalness"] > 0.5)
        .cast(pl.Int8)
        .alias("instrumentalness_binary")
    )

    numeric_df_1 = new_clean_numeric_df.drop("instrumentalness")
    return categorical_df_1, numeric_df_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.2.3 Normalization

    Many of the numeric features including danceability, energy, speechiness, acousticness, instrumentalness, liveness and valance are probalistic descriptors of a track. Because they are continuous measures intended to be compared directly, it is important that they share a common scale to avoid any single descriptor dominating the others. By Spotify's specification and from our own exploration we see that all of these features already fall within [0, 1] and threfore require no further action.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The remaining features, **loudness** and **tempo**, are also track descriptors but differ from the others in that they are direct measurements as opposed to probabilistic descriptors. Loudness is expressed on a dB scale (typically −60 to 0) and tempo ranges from approximately 50 to 250 BPM, so both require scaling to be comparable with the other features.

    To achieve this we apply Robust Scaling, which centres each feature on its median and scales by the interquartile range. This approach was chosen over Min-Max nomralization becaise, as observed in our earlier box plots, both loudness and tempo contain outliers. Min-Max is sensitive to outliers and would compress the majority of values into a narrow range. While Robust Scaling does not produce strict [0,1] bounds, for most models stable and representative scaling is preferable to a forced bounded range.
    """)
    return


@app.cell
def _(numeric_df_1, pl):
    features_to_scale = ["loudness", "tempo"]

    numeric_df_2 = numeric_df_1.with_columns([
        ((pl.col(c) - numeric_df_1[c].median()) / (numeric_df_1[c].quantile(0.75) - numeric_df_1[c].quantile(0.25)))
        .alias(c)
        for c in features_to_scale
    ])
    return (numeric_df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With tailored data preperation applied to each feature, we can analyse the final structure of our numeric data.
    """)
    return


@app.cell
def _(numeric_df_2, plt):
    pandas_numeric_df_2 = numeric_df_2.to_pandas()
    pandas_numeric_df_2.hist(bins=30, figsize=(12, 8))
    plt.tight_layout()
    plt.show()
    plt.close("")
    return (pandas_numeric_df_2,)


@app.cell
def _(pandas_numeric_df_2, plt):
    pandas_numeric_df_2.plot(
        kind="box",
        subplots=True,
        layout=(4, 4),   # adjust based on number of features
        figsize=(12, 10),
        sharex=False,
        sharey=False
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, pandas_numeric_df_2, plt, sns):
    corr_matrix = pandas_numeric_df_2.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Numeric Feature Correlation Matrix (Lower Triangle)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In our final exploration of the numeric data, popularity shows near-zero pairwise linear correlation with all audio features (all |r| ≤ 0.10). This suggests that no single audio feature is a strong linear predictor of popularity in isolation. However, a combination of features may still carry linear predictive signal.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 1.3 EDA of Categorical Data

    With our numeric attributes explored, we now move on to applying techniques to explore our categorical data
    """)
    return


@app.cell
def _(categorical_df_1):
    categorical_df_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first thing that sticks out when inspecting a portion of this categorical data is .....
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    #### 1.3.1 Genre Distribution
    """)
    return


@app.cell
def _(alt, clean_df):
    _genre_counts = clean_df.group_by("track_genre").len().rename({"len": "count"})

    eda_genre_chart = (
        alt.Chart(_genre_counts)
        .mark_bar()
        .encode(
            x=alt.X("track_genre:N", title="Genre", sort="-y"),
            y=alt.Y("count:Q", title="Number of Tracks"),
            color=alt.Color("track_genre:N", legend=None),
            tooltip=["track_genre", "count"],
        )
        .properties(title="Track Count per Genre", width=400, height=300)
    )
    eda_genre_chart
    return


@app.cell
def _(mo):
    mo.md("""
    The genre distribution is **imbalanced**: pop and indie-pop have the most tracks (~500 each),
    while r-n-b and hip-hop have fewer (~300 each). Synth-pop is in between (~400).
    This imbalance is relevant for clustering — if clusters aligned perfectly with genres,
    we would expect unequal cluster sizes. It also affects classification if genre is used as a feature.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    #### 1.3.2 Correlation Analysis
    """)
    return


@app.cell
def _(alt, clean_df, pd, pl):
    _corr_cols = [c for c in clean_df.columns if clean_df[c].dtype in (pl.Float64, pl.Int64, pl.Int32) and c != "track_genre"]
    _corr_pd = clean_df.select(_corr_cols).to_pandas().corr()
    _corr_melted = _corr_pd.reset_index().melt(id_vars="index")
    _corr_melted.columns = ["Feature 1", "Feature 2", "Correlation"]

    eda_corr_chart = (
        alt.Chart(pd.DataFrame(_corr_melted))
        .mark_rect()
        .encode(
            x=alt.X("Feature 1:N", title=""),
            y=alt.Y("Feature 2:N", title=""),
            color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["Feature 1", "Feature 2", alt.Tooltip("Correlation:Q", format=".2f")],
        )
        .properties(title="Feature Correlation Matrix", width=500, height=500)
    )
    eda_corr_chart
    return


@app.cell
def _(mo):
    mo.md("""
    Key correlations:

    - **Energy and loudness** have a strong positive correlation (0.633) — louder tracks tend to be more energetic
    - **Acousticness** is negatively correlated with both energy and loudness — acoustic tracks are quieter and calmer
    - **Danceability and valence** show moderate positive correlation — happier-sounding tracks tend to be more danceable
    - **Popularity shows almost no linear correlation with any individual audio feature** (all |r| ≤ 0.10). This means linear models will struggle — any predictive power must come from non-linear interactions or genre.

    These correlations matter for modelling: energy-loudness multicollinearity could affect linear models (logistic regression, ridge).
    Tree-based models are robust to this. We will use StandardScaler inside pipelines, which does not eliminate collinearity
    but ensures no feature dominates due to scale.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    #### 1.3.3 Popularity by Genre
    """)
    return


@app.cell
def _(alt, clean_df):
    eda_pop_genre_chart = (
        alt.Chart(clean_df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("track_genre:N", title="Genre", sort="-y"),
            y=alt.Y("popularity:Q", title="Popularity"),
            color=alt.Color("track_genre:N", legend=None),
        )
        .properties(title="Popularity Distribution by Genre", width=400, height=300)
    )
    eda_pop_genre_chart
    return


@app.cell
def _(mo):
    mo.md("""
    **Synth-pop** has the lowest median popularity (29), while **pop** has the highest (66).
    Hip-hop (58), indie-pop (47), and r-n-b (42) fall in between.
    The large number of zero-popularity tracks appears across all genres, suggesting it is a data artifact
    (unlisted or removed tracks) rather than a genre-specific phenomenon.
    Genre carries some predictive signal for popularity, which justifies including it as a feature in classification and regression.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    #### 1.3.4 Feature Distributions by Genre
    """)
    return


@app.cell
def _(clean_df, mo, pl):
    _feature_options = [c for c in clean_df.columns if clean_df[c].dtype in (pl.Float64, pl.Int64, pl.Int32) and c not in ("popularity", "track_genre")]
    eda_feature_selector = mo.ui.dropdown(
        options=_feature_options,
        value="danceability",
        label="Select Feature:",
    )
    eda_feature_selector
    return (eda_feature_selector,)


@app.cell
def _(alt, clean_df, eda_feature_selector):
    eda_feature_genre_chart = (
        alt.Chart(clean_df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("track_genre:N", title="Genre"),
            y=alt.Y(f"{eda_feature_selector.value}:Q", title=eda_feature_selector.value),
            color=alt.Color("track_genre:N", legend=None),
        )
        .properties(
            title=f"{eda_feature_selector.value} by Genre",
            width=400,
            height=300,
        )
    )
    eda_feature_genre_chart
    return


@app.cell
def _(mo):
    mo.md("""
    Using the dropdown above, we can explore how each audio feature varies across genres:

    - **Speechiness** is distinctly higher for hip-hop (vocal-heavy genre)
    - **Danceability** is higher for r-n-b and hip-hop
    - **Acousticness** varies widely across genres
    - **Energy** distributions are fairly similar across genres, making it a weaker genre discriminator
    - Synth-pop and indie-pop overlap on many features, which may make them hard to separate in clustering
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 1.4 EDA Summary — Implications for Downstream Tasks

    **Key finding:** Popularity has near-zero linear correlation with every audio feature (all |r| ≤ 0.10).
    No single feature is a good linear predictor of popularity. This will be the dominant theme across all modelling tasks.

    **For Clustering:** Audio features show some genre-related patterns (e.g., hip-hop's high speechiness)
    but also significant overlap between genres like synth-pop and indie-pop. We expect clustering to
    partially recover genre structure but not achieve clean separation. Standardisation is essential
    because features like `duration_ms` (~200000) would dominate distance-based algorithms.

    **For Classification:** The popularity distribution suggests a roughly balanced binary split around the median.
    Since no single audio feature correlates with popularity, ensemble models that capture non-linear interactions
    will be essential. Logistic regression is expected to perform poorly.
    However, popularity is driven by factors outside the dataset (marketing, artist fame), so accuracy will be limited.

    **For Regression:** The spike at zero popularity and the right-skewed distribution will make exact prediction
    very difficult. The near-zero correlations mean R-squared for linear models will be extremely low.
    The zero-popularity tracks add noise that hurts regression more than classification
    (since most zeros fall below the median anyway, classification correctly assigns them to class 0).



    the dataset structures, distributions, and relationships present in the dataset. Once we have established the primary charachteristics of the dataset and concluded our inferences from our observations we will use these inferences to inform our approaces to clustering, classification and regression.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Clustering (Descriptive Analytics)

    2. **Clustering** — discovering natural groupings using K-Means and DBSCAN
    3. **Classification** — predicting whether a track is above or below median popularity
    4. **Regression** — predicting the exact popularity score
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2.1 Preparing Features for Clustering
    """)
    return


@app.cell
def _(clean_df, mo):
    cl_feature_cols = [c for c in clean_df.columns if c not in ("track_genre", "popularity")]
    cl_X = clean_df.select(cl_feature_cols).to_pandas().values
    cl_genres = clean_df["track_genre"].to_list()

    mo.md(f"""
    **Clustering features ({len(cl_feature_cols)}):** {', '.join(cl_feature_cols)}

    We drop `track_genre` as required by the spec (clustering is unsupervised — we want to discover structure
    from audio features alone). We also drop `popularity` because including it would bias clusters toward
    popularity-based groupings rather than audio-feature-based groupings; our goal is to see whether audio
    features reveal genre structure.

    **Preprocessing justification:** We use `StandardScaler` inside all clustering pipelines because K-Means
    uses Euclidean distance, which is sensitive to scale. Without scaling, `duration_ms` (~200000) would
    dominate over features like `danceability` (~0.7). DBSCAN's `eps` parameter also depends on distance scale.
    We do not apply PCA as a preprocessing step — PCA would reduce interpretability and is only used later for
    2D visualisation. We do not log-transform skewed features because StandardScaler is sufficient for
    distance-based algorithms and preserves the original feature semantics.
    """)
    return cl_X, cl_genres


@app.cell
def _(mo):
    mo.md("""
    ### 2.2 K-Means: Elbow Method
    """)
    return


@app.cell
def _(KMeans, Pipeline, StandardScaler, alt, cl_X, pl):
    km_k_range = range(2, 16)
    km_inertias = []
    for _k in km_k_range:
        _pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=_k, random_state=42, n_init=10)),
        ])
        _pipe.fit(cl_X)
        km_inertias.append({"k": _k, "inertia": _pipe.named_steps["kmeans"].inertia_})

    _elbow_df = pl.DataFrame(km_inertias)

    km_elbow_chart = (
        alt.Chart(_elbow_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="Number of Clusters (k)", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("inertia:Q", title="Inertia (Within-Cluster Sum of Squares)"),
            tooltip=["k", alt.Tooltip("inertia:Q", format=".0f")],
        )
        .properties(title="K-Means Elbow Method", width=500, height=300)
    )
    km_elbow_chart
    return (km_k_range,)


@app.cell
def _(mo):
    mo.md("""
    The elbow plot shows diminishing returns in inertia reduction after approximately **k=5**, which aligns with
    the 5 genres in our dataset. However, the elbow is not sharp — inertia decreases gradually, suggesting
    the data does not form highly distinct clusters in audio feature space. This is consistent with our EDA
    finding that genres overlap on many features.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2.3 K-Means: Silhouette Analysis
    """)
    return


@app.cell
def _(
    KMeans,
    Pipeline,
    StandardScaler,
    alt,
    cl_X,
    km_k_range,
    pl,
    silhouette_score,
):
    km_silhouettes = []
    for _k in km_k_range:
        _pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=_k, random_state=42, n_init=10)),
        ])
        _pipe.fit(cl_X)
        _labels = _pipe.named_steps["kmeans"].labels_
        _X_scaled = _pipe.named_steps["scaler"].transform(cl_X)
        _sil = silhouette_score(_X_scaled, _labels)
        km_silhouettes.append({"k": _k, "silhouette": round(_sil, 4)})

    _sil_df = pl.DataFrame(km_silhouettes)

    km_sil_chart = (
        alt.Chart(_sil_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="Number of Clusters (k)", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("silhouette:Q", title="Silhouette Score"),
            tooltip=["k", alt.Tooltip("silhouette:Q", format=".4f")],
        )
        .properties(title="Silhouette Score by k", width=500, height=300)
    )
    km_sil_chart
    return (km_silhouettes,)


@app.cell
def _(km_silhouettes, mo):
    _best_sil = max(km_silhouettes, key=lambda x: x["silhouette"])
    mo.md(f"""
    The highest silhouette score is **{_best_sil['silhouette']:.4f}** at **k={_best_sil['k']}**, but the scores
    are very flat (ranging from ~0.12 to ~0.16), meaning no k value produces well-separated clusters.
    All scores are below 0.2, indicating substantial overlap between clusters regardless of k.
    The fact that k=5 (matching the genre count) does not yield the best silhouette score tells us that
    genre boundaries in audio feature space are blurry — genres are cultural categories, not purely acoustic ones.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2.4 K-Means: Comparing k=3 and k=5
    """)
    return


@app.cell
def _(KMeans, PCA, Pipeline, StandardScaler, alt, cl_X, cl_genres, mo, pl):
    km_pipe_3 = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10)),
    ])
    km_pipe_3.fit(cl_X)
    km_labels_3 = km_pipe_3.named_steps["kmeans"].labels_

    km_pipe_5 = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=5, random_state=42, n_init=10)),
    ])
    km_pipe_5.fit(cl_X)
    km_labels_5 = km_pipe_5.named_steps["kmeans"].labels_

    km_pca = PCA(n_components=2)
    _X_scaled = km_pipe_5.named_steps["scaler"].transform(cl_X)
    km_pca_coords = km_pca.fit_transform(_X_scaled)

    _viz_df_3 = pl.DataFrame({
        "PC1": km_pca_coords[:, 0],
        "PC2": km_pca_coords[:, 1],
        "Cluster": [str(c) for c in km_labels_3],
        "Genre": cl_genres,
    })
    _viz_df_5 = pl.DataFrame({
        "PC1": km_pca_coords[:, 0],
        "PC2": km_pca_coords[:, 1],
        "Cluster": [str(c) for c in km_labels_5],
        "Genre": cl_genres,
    })

    _chart_3 = (
        alt.Chart(_viz_df_3)
        .mark_circle(size=30, opacity=0.5)
        .encode(
            x=alt.X("PC1:Q", title="PC1"),
            y=alt.Y("PC2:Q", title="PC2"),
            color=alt.Color("Cluster:N", title="Cluster"),
            tooltip=["Genre", "Cluster"],
        )
        .properties(title="K-Means k=3", width=350, height=300)
    )
    _chart_5 = (
        alt.Chart(_viz_df_5)
        .mark_circle(size=30, opacity=0.5)
        .encode(
            x=alt.X("PC1:Q", title="PC1"),
            y=alt.Y("PC2:Q", title="PC2"),
            color=alt.Color("Cluster:N", title="Cluster"),
            tooltip=["Genre", "Cluster"],
        )
        .properties(title="K-Means k=5", width=350, height=300)
    )

    km_viz_chart = mo.hstack([_chart_3, _chart_5])
    km_viz_chart
    return km_labels_3, km_labels_5, km_pca, km_pca_coords


@app.cell
def _(km_labels_3, km_labels_5, km_pca, km_pca_coords, mo, silhouette_score):
    _sil_3 = silhouette_score(km_pca_coords, km_labels_3)
    _sil_5 = silhouette_score(km_pca_coords, km_labels_5)
    mo.md(f"""
    PCA projects the 14-dimensional feature space to 2D for visualisation (explaining
    {km_pca.explained_variance_ratio_.sum() * 100:.1f}% of variance). Both k=3 and k=5 show
    overlapping clusters, confirming that audio features do not form tight, well-separated groups.

    - **k=3** produces broader clusters (silhouette: {_sil_3:.4f}) — coherent but cannot map 1:1 to 5 genres
    - **k=5** (silhouette: {_sil_5:.4f}) attempts to match genre count but clusters still overlap heavily

    We will compare cluster assignments to true genres next.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2.5 K-Means: Cluster vs Genre Alignment
    """)
    return


@app.cell
def _(alt, cl_genres, km_labels_5, pl):
    _ct_df = pl.DataFrame({"Cluster": [str(c) for c in km_labels_5], "Genre": cl_genres})
    _ct = _ct_df.group_by(["Cluster", "Genre"]).len().rename({"len": "Count"})

    km_align_chart = (
        alt.Chart(_ct)
        .mark_rect()
        .encode(
            x=alt.X("Cluster:N", title="K-Means Cluster (k=5)"),
            y=alt.Y("Genre:N", title="True Genre"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Cluster", "Genre", "Count"],
        )
        .properties(title="K-Means Cluster vs True Genre (k=5)", width=350, height=250)
    )
    km_align_chart
    return


@app.cell
def _(mo):
    mo.md("""
    The cross-tabulation reveals that K-Means clusters do **not** cleanly correspond to genres.
    Hip-hop is the most concentrated genre, clustering heavily in one group (due to its distinctive high speechiness).
    Synth-pop also concentrates in a single cluster. However, pop and indie-pop tracks are spread across
    all clusters, consistent with their overlapping audio profiles found in EDA.
    This confirms that audio features capture some genre signal but genres are not
    purely defined by numeric features — lyrics, artist identity, and cultural context matter too.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2.6 DBSCAN Clustering
    """)
    return


@app.cell
def _(DBSCAN, Pipeline, StandardScaler, cl_X, cl_genres, pl, silhouette_score):
    db_results = []
    for _eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for _ms in [3, 5, 10]:
            _pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("dbscan", DBSCAN(eps=_eps, min_samples=_ms)),
            ])
            _pipe.fit(cl_X)
            _labels = _pipe.named_steps["dbscan"].labels_
            _n_clusters = len(set(_labels)) - (1 if -1 in _labels else 0)
            _n_noise = list(_labels).count(-1)
            _sil = None
            if _n_clusters >= 2 and _n_noise < len(_labels) * 0.5:
                _non_noise = [i for i, l in enumerate(_labels) if l != -1]
                if len(set(_labels[i] for i in _non_noise)) >= 2:
                    _X_scaled = _pipe.named_steps["scaler"].transform(cl_X)
                    _sil = round(silhouette_score(
                        _X_scaled[_non_noise],
                        [_labels[i] for i in _non_noise],
                    ), 4)
            db_results.append({
                "eps": _eps,
                "min_samples": _ms,
                "n_clusters": _n_clusters,
                "n_noise": _n_noise,
                "noise_pct": round(_n_noise / len(cl_genres) * 100, 1),
                "silhouette": _sil,
            })

    db_results_df = pl.DataFrame(db_results)
    db_results_df
    return (db_results,)


@app.cell
def _(mo):
    mo.md("""
    DBSCAN struggles with this dataset. At low `eps` values, most points are classified as noise.
    At higher `eps` values (e.g., eps=3.0), DBSCAN does find some structure with low noise — but the
    neighbourhood radius is very large, meaning clusters are loose. Silhouette scores are only shown
    when less than 50% of points are noise, since tiny clusters amid massive noise produce
    misleadingly high silhouette scores (0.99+). At eps=3.0, the silhouette (~0.19) is comparable
    to K-Means (~0.12-0.16), but neither result indicates well-separated clusters.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2.7 DBSCAN: Best Configuration
    """)
    return


@app.cell
def _(
    DBSCAN,
    Pipeline,
    StandardScaler,
    alt,
    cl_X,
    cl_genres,
    db_results,
    km_pca,
    pl,
):
    _valid = [r for r in db_results if r["n_clusters"] >= 2 and r["silhouette"] is not None]
    if _valid:
        _best_db = max(_valid, key=lambda r: r["silhouette"])
        db_best_eps = _best_db["eps"]
        db_best_ms = _best_db["min_samples"]
    else:
        db_best_eps = 2.0
        db_best_ms = 5

    db_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("dbscan", DBSCAN(eps=db_best_eps, min_samples=db_best_ms)),
    ])
    db_pipe.fit(cl_X)
    db_labels = db_pipe.named_steps["dbscan"].labels_

    _X_scaled_db = db_pipe.named_steps["scaler"].transform(cl_X)
    _db_pca_coords = km_pca.transform(_X_scaled_db)

    _db_n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    _db_n_noise = list(db_labels).count(-1)

    _db_viz = pl.DataFrame({
        "PC1": _db_pca_coords[:, 0],
        "PC2": _db_pca_coords[:, 1],
        "Cluster": [str(c) for c in db_labels],
        "Genre": cl_genres,
    })

    db_viz_chart = (
        alt.Chart(_db_viz)
        .mark_circle(size=30, opacity=0.5)
        .encode(
            x=alt.X("PC1:Q", title="PC1"),
            y=alt.Y("PC2:Q", title="PC2"),
            color=alt.Color("Cluster:N", title="Cluster"),
            tooltip=["Genre", "Cluster"],
        )
        .properties(
            title=f"DBSCAN (eps={db_best_eps}, min_samples={db_best_ms}) — {_db_n_clusters} clusters, {_db_n_noise} noise points",
            width=500,
            height=350,
        )
    )
    db_viz_chart
    return (db_labels,)


@app.cell
def _(mo):
    mo.md("""
    ### 2.8 DBSCAN: Cluster vs Genre Alignment
    """)
    return


@app.cell
def _(alt, cl_genres, db_labels, pl):
    _db_ct = pl.DataFrame({"Cluster": [str(c) for c in db_labels], "Genre": cl_genres})
    _db_ct_agg = _db_ct.group_by(["Cluster", "Genre"]).len().rename({"len": "Count"})

    db_align_chart = (
        alt.Chart(_db_ct_agg)
        .mark_rect()
        .encode(
            x=alt.X("Cluster:N", title="DBSCAN Cluster (-1 = noise)"),
            y=alt.Y("Genre:N", title="True Genre"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Cluster", "Genre", "Count"],
        )
        .properties(title="DBSCAN Cluster vs True Genre", width=350, height=250)
    )
    db_align_chart
    return


@app.cell
def _(mo):
    mo.md("""
    DBSCAN's genre alignment depends heavily on the eps value. At low eps, most tracks are noise.
    At the best eps (3.0), DBSCAN finds a few clusters with low noise, but the large neighbourhood radius
    means clusters are loose and do not correspond cleanly to genres. K-Means provides more interpretable
    genre alignment, particularly for hip-hop and synth-pop which concentrate in distinct clusters.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2.9 Clustering Conclusion

    Both algorithms produce weak clusters (silhouette scores below 0.2). We select **K-Means with k=5** as our final clustering solution for the following reasons:

    1. **k=5 matches the known genre count**, and while clusters do not cleanly map to genres, partial structure exists
       (e.g., hip-hop concentrates in one cluster due to high speechiness, synth-pop concentrates in another)
    2. **The elbow method** shows diminishing returns after k=5, confirming it as a reasonable choice
    3. **DBSCAN at eps=3.0** finds similar structure with low noise, but the large neighbourhood radius means
       clusters are loose. Its best silhouette (~0.19) is comparable to K-Means (~0.16) — neither is clearly better.
       At low eps, most points are noise, making DBSCAN impractical.

    **Why clustering only partially succeeded:** Genres are cultural categories defined by more than audio features.
    Synth-pop and indie-pop share similar audio profiles, making them hard to separate.
    Hip-hop's distinctively high speechiness makes it the most separable genre.
    Overall, audio features provide a weak but detectable signal for genre grouping.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Classification (Predicting Popularity Category)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3.1 Creating the Binary Target
    """)
    return


@app.cell
def _(clean_df, mo, pl):
    clf_median = clean_df["popularity"].median()
    clf_df = clean_df.with_columns(
        pl.when(pl.col("popularity") > clf_median).then(1).otherwise(0).alias("popularity_binary")
    ).drop("popularity")

    _class_counts = clf_df.group_by("popularity_binary").len().rename({"len": "count"}).sort("popularity_binary")
    mo.vstack([
        mo.md(f"**Median popularity:** {clf_median}"),
        mo.md(f"**Binary target:** 0 if popularity <= {clf_median}, 1 if popularity > {clf_median}"),
        _class_counts,
        mo.md(f"**Class balance:** {_class_counts['count'][0]} (class 0) vs {_class_counts['count'][1]} (class 1) — {'balanced' if abs(_class_counts['count'][0] - _class_counts['count'][1]) / _class_counts['count'].sum() < 0.1 else 'slightly imbalanced'}"),
    ])
    return (clf_df,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.2 Preparing Data and Pipeline
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    clf_df,
    mo,
    train_test_split,
):
    _clf_pdf = clf_df.to_pandas()
    clf_numeric_cols = [c for c in _clf_pdf.columns if c not in ("track_genre", "popularity_binary")]
    clf_cat_cols = ["track_genre"]

    clf_X = _clf_pdf.drop(columns=["popularity_binary"])
    clf_y = _clf_pdf["popularity_binary"].values

    clf_X_train, clf_X_test, clf_y_train, clf_y_test = train_test_split(
        clf_X, clf_y, test_size=0.2, random_state=42, stratify=clf_y
    )

    clf_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), clf_numeric_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), clf_cat_cols),
    ])

    mo.md(f"""
    **Features:** {len(clf_numeric_cols)} numeric + 1 categorical (`track_genre`, one-hot encoded)

    **Pipeline structure:** `ColumnTransformer` (StandardScaler for numeric, OneHotEncoder for genre) → Model.
    All preprocessing is inside the pipeline, preventing data leakage — the scaler fits only on training folds during CV.

    **Train/test split:** 80/20, stratified by target class, `random_state=42` for reproducibility.
    We use stratified splitting because even though classes are roughly balanced by construction (median split),
    small imbalances should be preserved proportionally in train and test sets.

    **Cross-validation:** 5-fold CV on the training set to estimate generalisation performance before final evaluation on the held-out test set.

    **Train set:** {clf_X_train.shape[0]} samples | **Test set:** {clf_X_test.shape[0]} samples
    """)
    return clf_X_test, clf_X_train, clf_preprocessor, clf_y_test, clf_y_train


@app.cell
def _(mo):
    mo.md("""
    ### 3.3 Evaluation Metric Justification

    We use **accuracy** as our primary metric for model selection because the classes are approximately balanced
    after the median split. With balanced classes, accuracy is an appropriate and interpretable metric — it directly
    answers "what fraction of tracks did the model classify correctly?" We also report precision, recall, and F1-score
    for completeness on the final model, as these provide insight into false positive vs false negative trade-offs.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3.4 Model 1: Logistic Regression
    """)
    return


@app.cell
def _(
    LogisticRegression,
    Pipeline,
    clf_X_train,
    clf_preprocessor,
    clf_y_train,
    cross_val_score,
    mo,
):
    clf_lr_pipe = Pipeline([
        ("preprocessor", clf_preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf_lr_cv = cross_val_score(clf_lr_pipe, clf_X_train, clf_y_train, cv=5, scoring="accuracy")

    mo.md(f"""
    **Logistic Regression** — a linear baseline model.

    5-fold CV accuracy: **{clf_lr_cv.mean():.4f}** (+/- {clf_lr_cv.std():.4f})

    As a linear model, logistic regression assumes the decision boundary between popular and unpopular tracks
    is a hyperplane in feature space. If the true boundary is non-linear, tree-based models should outperform this.
    """)
    return clf_lr_cv, clf_lr_pipe


@app.cell
def _(mo):
    mo.md("""
    ### 3.5 Model 2: Random Forest
    """)
    return


@app.cell
def _(
    Pipeline,
    RandomForestClassifier,
    clf_X_train,
    clf_preprocessor,
    clf_y_train,
    cross_val_score,
    mo,
):
    clf_rf_pipe = Pipeline([
        ("preprocessor", clf_preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    clf_rf_cv = cross_val_score(clf_rf_pipe, clf_X_train, clf_y_train, cv=5, scoring="accuracy")

    mo.md(f"""
    **Random Forest** — an ensemble of decision trees that can capture non-linear relationships.

    5-fold CV accuracy: **{clf_rf_cv.mean():.4f}** (+/- {clf_rf_cv.std():.4f})

    Random Forest handles feature interactions and non-linearities that logistic regression cannot.
    It also provides feature importance rankings via the Gini impurity criterion.
    """)
    return clf_rf_cv, clf_rf_pipe


@app.cell
def _(mo):
    mo.md("""
    ### 3.6 Model 3: Gradient Boosting
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    Pipeline,
    clf_X_train,
    clf_preprocessor,
    clf_y_train,
    cross_val_score,
    mo,
):
    clf_gb_pipe = Pipeline([
        ("preprocessor", clf_preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ])
    clf_gb_cv = cross_val_score(clf_gb_pipe, clf_X_train, clf_y_train, cv=5, scoring="accuracy")

    mo.md(f"""
    **Gradient Boosting** — builds trees sequentially, each correcting the errors of the previous one.

    5-fold CV accuracy: **{clf_gb_cv.mean():.4f}** (+/- {clf_gb_cv.std():.4f})

    Gradient boosting often achieves the best performance on tabular data because it focuses on hard-to-classify
    samples iteratively. However, it is more prone to overfitting than random forest if not regularised.
    """)
    return clf_gb_cv, clf_gb_pipe


@app.cell
def _(mo):
    mo.md("""
    ### 3.7 Classification Model Comparison
    """)
    return


@app.cell
def _(alt, clf_gb_cv, clf_lr_cv, clf_rf_cv, mo, pl):
    _clf_comp = pl.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "CV Accuracy (mean)": [round(clf_lr_cv.mean(), 4), round(clf_rf_cv.mean(), 4), round(clf_gb_cv.mean(), 4)],
        "CV Accuracy (std)": [round(clf_lr_cv.std(), 4), round(clf_rf_cv.std(), 4), round(clf_gb_cv.std(), 4)],
    })

    clf_comparison_chart = (
        alt.Chart(_clf_comp)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title="", sort="-y"),
            y=alt.Y("CV Accuracy (mean):Q", title="Mean CV Accuracy", scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", "CV Accuracy (mean)", "CV Accuracy (std)"],
        )
        .properties(title="Classification Model Comparison (5-Fold CV)", width=400, height=300)
    )

    mo.vstack([clf_comparison_chart, _clf_comp])
    return


@app.cell
def _(clf_gb_cv, clf_lr_cv, clf_rf_cv, mo):
    _best_clf = max(
        [("Logistic Regression", clf_lr_cv.mean()), ("Random Forest", clf_rf_cv.mean()), ("Gradient Boosting", clf_gb_cv.mean())],
        key=lambda x: x[1],
    )
    mo.md(f"""
    **{_best_clf[0]}** achieves the highest cross-validated accuracy ({_best_clf[1]:.4f}).
    Tree-based models (Random Forest, Gradient Boosting) significantly outperform logistic regression,
    confirming that the decision boundary is non-linear — consistent with the near-zero linear correlations
    found in EDA. ~70% accuracy is modest but meaningful, beating random guessing (50%) by 20 percentage points.
    Performance is limited because popularity depends heavily on marketing, artist fame,
    playlist placement, and release timing, none of which are captured in our features.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3.8 Final Classification Model: Test Set Evaluation
    """)
    return


@app.cell
def _(
    alt,
    classification_report,
    clf_X_test,
    clf_X_train,
    clf_gb_cv,
    clf_gb_pipe,
    clf_lr_cv,
    clf_lr_pipe,
    clf_rf_cv,
    clf_rf_pipe,
    clf_y_test,
    clf_y_train,
    confusion_matrix,
    mo,
    pd,
):
    _cv_scores = {"Logistic Regression": clf_lr_cv.mean(), "Random Forest": clf_rf_cv.mean(), "Gradient Boosting": clf_gb_cv.mean()}
    _best_name = max(_cv_scores, key=_cv_scores.get)
    _pipe_map = {"Logistic Regression": clf_lr_pipe, "Random Forest": clf_rf_pipe, "Gradient Boosting": clf_gb_pipe}
    clf_final_pipe = _pipe_map[_best_name]

    clf_final_pipe.fit(clf_X_train, clf_y_train)
    clf_final_pred = clf_final_pipe.predict(clf_X_test)
    clf_test_acc = (clf_final_pred == clf_y_test).mean()

    _cm = confusion_matrix(clf_y_test, clf_final_pred)
    _cm_df = pd.DataFrame(
        _cm,
        index=["Actual: Low", "Actual: High"],
        columns=["Pred: Low", "Pred: High"],
    )
    _cm_melted = _cm_df.reset_index().melt(id_vars="index")
    _cm_melted.columns = ["Actual", "Predicted", "Count"]

    clf_cm_chart = (
        alt.Chart(pd.DataFrame(_cm_melted))
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", title="Predicted"),
            y=alt.Y("Actual:N", title="Actual"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Actual", "Predicted", "Count"],
        )
        .mark_rect()
        .encode(
            x="Predicted:N",
            y="Actual:N",
            color="Count:Q",
        )
        .properties(title=f"Confusion Matrix — {_best_name}", width=300, height=250)
    )

    clf_report = classification_report(clf_y_test, clf_final_pred, target_names=["Low", "High"])

    mo.vstack([
        mo.md(f"**Final model:** {_best_name} | **Test accuracy:** {clf_test_acc:.4f}"),
        clf_cm_chart,
        mo.md(f"```\n{clf_report}\n```"),
    ])
    return (clf_final_pipe,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.9 Feature Importance for Classification
    """)
    return


@app.cell
def _(alt, clf_final_pipe, np, pl):
    _clf_model = clf_final_pipe.named_steps["clf"]
    if hasattr(_clf_model, "feature_importances_"):
        _importances = _clf_model.feature_importances_
    else:
        _importances = np.abs(_clf_model.coef_[0])

    _feat_names = clf_final_pipe.named_steps["preprocessor"].get_feature_names_out()
    _feat_names = [f.split("__")[-1] for f in _feat_names]

    _imp_df = pl.DataFrame({"Feature": _feat_names, "Importance": _importances}).sort("Importance", descending=True)

    clf_feat_imp_chart = (
        alt.Chart(_imp_df.head(15))
        .mark_bar()
        .encode(
            x=alt.X("Importance:Q", title="Feature Importance"),
            y=alt.Y("Feature:N", sort="-x", title=""),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(title="Top 15 Feature Importances (Classification)", width=450, height=350)
    )
    clf_feat_imp_chart
    return


@app.cell
def _(mo):
    mo.md("""
    **No single feature dominates** — all audio features contribute roughly equally (6-10% importance each).
    The top features include acousticness, tempo, danceability, and speechiness.
    Interestingly, genre features rank lower than audio features, suggesting that a track's audio profile
    matters more than its genre label for predicting popularity. This is somewhat surprising given that
    genre medians differ significantly, but it aligns with the finding that popularity has no strong
    linear correlation with any individual feature — the model relies on interactions between many features.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Regression (Predicting Popularity Score)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.1 Preparing Data for Regression
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    clean_df,
    mo,
    train_test_split,
):
    reg_df = clean_df

    _reg_pdf = reg_df.to_pandas()
    reg_numeric_cols = [c for c in _reg_pdf.columns if c not in ("track_genre", "popularity")]
    reg_cat_cols = ["track_genre"]

    reg_X = _reg_pdf.drop(columns=["popularity"])
    reg_y = _reg_pdf["popularity"].values

    reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(
        reg_X, reg_y, test_size=0.2, random_state=42
    )

    reg_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), reg_numeric_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), reg_cat_cols),
    ])

    mo.md(f"""
    We use a **separate copy** of the cleaned dataset with the original `popularity` column retained
    (not binarised), as required by the spec.

    **Pipeline structure:** Same `ColumnTransformer` pattern as classification (StandardScaler for numeric,
    OneHotEncoder for genre) → Regression model. All preprocessing inside the pipeline.

    **Train/test split:** 80/20, `random_state=42`. Not stratified (continuous target).

    **Train set:** {reg_X_train.shape[0]} samples | **Test set:** {reg_X_test.shape[0]} samples
    """)
    return reg_X_test, reg_X_train, reg_preprocessor, reg_y_test, reg_y_train


@app.cell
def _(mo):
    mo.md("""
    ### 4.2 Evaluation Metric Justification

    We use **RMSE** (Root Mean Squared Error) as our primary metric because it is in the same units as
    the target (popularity points), making it directly interpretable — e.g., "the model is off by ~X points on average."
    RMSE penalises large errors more heavily than MAE, which is appropriate here because being off by 50 points
    is much worse than being off by 5 points twice.

    We also report **R-squared** (coefficient of determination) to quantify what fraction of popularity variance
    our features explain, providing context for how much signal exists in the data.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.3 Model 1: Ridge Regression
    """)
    return


@app.cell
def _(
    Pipeline,
    Ridge,
    cross_val_score,
    mo,
    np,
    reg_X_train,
    reg_preprocessor,
    reg_y_train,
):
    reg_ridge_pipe = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("reg", Ridge(alpha=1.0)),
    ])
    reg_ridge_cv_mse = cross_val_score(reg_ridge_pipe, reg_X_train, reg_y_train, cv=5, scoring="neg_mean_squared_error")
    reg_ridge_cv_r2 = cross_val_score(reg_ridge_pipe, reg_X_train, reg_y_train, cv=5, scoring="r2")

    _rmse = np.sqrt(-reg_ridge_cv_mse.mean())

    mo.md(f"""
    **Ridge Regression** — linear regression with L2 regularisation to handle multicollinearity (energy-loudness correlation found in EDA).

    5-fold CV RMSE: **{_rmse:.2f}** | R-squared: **{reg_ridge_cv_r2.mean():.4f}** (+/- {reg_ridge_cv_r2.std():.4f})

    The R-squared is near zero — essentially no predictive power from a linear model. This confirms
    the EDA finding: there is no linear relationship between audio features and popularity.
    RMSE of ~29 points (out of a 0-100 range) means predictions are off by nearly a third of the scale.
    """)
    return reg_ridge_cv_mse, reg_ridge_cv_r2, reg_ridge_pipe


@app.cell
def _(mo):
    mo.md("""
    ### 4.4 Model 2: Random Forest Regressor
    """)
    return


@app.cell
def _(
    Pipeline,
    RandomForestRegressor,
    cross_val_score,
    mo,
    np,
    reg_X_train,
    reg_preprocessor,
    reg_y_train,
):
    reg_rf_pipe = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("reg", RandomForestRegressor(n_estimators=200, random_state=42)),
    ])
    reg_rf_cv_mse = cross_val_score(reg_rf_pipe, reg_X_train, reg_y_train, cv=5, scoring="neg_mean_squared_error")
    reg_rf_cv_r2 = cross_val_score(reg_rf_pipe, reg_X_train, reg_y_train, cv=5, scoring="r2")

    _rmse = np.sqrt(-reg_rf_cv_mse.mean())

    mo.md(f"""
    **Random Forest Regressor** — ensemble of decision trees for non-linear regression.

    5-fold CV RMSE: **{_rmse:.2f}** | R-squared: **{reg_rf_cv_r2.mean():.4f}** (+/- {reg_rf_cv_r2.std():.4f})
    """)
    return reg_rf_cv_mse, reg_rf_cv_r2, reg_rf_pipe


@app.cell
def _(mo):
    mo.md("""
    ### 4.5 Model 3: Gradient Boosting Regressor
    """)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    Pipeline,
    cross_val_score,
    mo,
    np,
    reg_X_train,
    reg_preprocessor,
    reg_y_train,
):
    reg_gb_pipe = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("reg", GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ])
    reg_gb_cv_mse = cross_val_score(reg_gb_pipe, reg_X_train, reg_y_train, cv=5, scoring="neg_mean_squared_error")
    reg_gb_cv_r2 = cross_val_score(reg_gb_pipe, reg_X_train, reg_y_train, cv=5, scoring="r2")

    _rmse = np.sqrt(-reg_gb_cv_mse.mean())

    mo.md(f"""
    **Gradient Boosting Regressor** — sequential boosting for regression.

    5-fold CV RMSE: **{_rmse:.2f}** | R-squared: **{reg_gb_cv_r2.mean():.4f}** (+/- {reg_gb_cv_r2.std():.4f})
    """)
    return reg_gb_cv_mse, reg_gb_cv_r2, reg_gb_pipe


@app.cell
def _(mo):
    mo.md("""
    ### 4.6 Regression Model Comparison
    """)
    return


@app.cell
def _(
    alt,
    mo,
    np,
    pl,
    reg_gb_cv_mse,
    reg_gb_cv_r2,
    reg_rf_cv_mse,
    reg_rf_cv_r2,
    reg_ridge_cv_mse,
    reg_ridge_cv_r2,
):
    _reg_comp = pl.DataFrame({
        "Model": ["Ridge", "Random Forest", "Gradient Boosting"],
        "CV RMSE": [
            round(np.sqrt(-reg_ridge_cv_mse.mean()), 2),
            round(np.sqrt(-reg_rf_cv_mse.mean()), 2),
            round(np.sqrt(-reg_gb_cv_mse.mean()), 2),
        ],
        "CV R-squared": [
            round(reg_ridge_cv_r2.mean(), 4),
            round(reg_rf_cv_r2.mean(), 4),
            round(reg_gb_cv_r2.mean(), 4),
        ],
    })

    _rmse_chart = (
        alt.Chart(_reg_comp)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title="", sort="y"),
            y=alt.Y("CV RMSE:Q", title="CV RMSE (lower is better)"),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", "CV RMSE", "CV R-squared"],
        )
        .properties(title="Regression: RMSE Comparison", width=250, height=250)
    )
    _r2_chart = (
        alt.Chart(_reg_comp)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title="", sort="-y"),
            y=alt.Y("CV R-squared:Q", title="CV R-squared (higher is better)"),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", "CV RMSE", "CV R-squared"],
        )
        .properties(title="Regression: R-squared Comparison", width=250, height=250)
    )

    reg_comparison_chart = mo.hstack([_rmse_chart, _r2_chart])

    mo.vstack([reg_comparison_chart, _reg_comp])
    return


@app.cell
def _(mo, reg_gb_cv_r2, reg_rf_cv_r2, reg_ridge_cv_r2):
    _best_reg = max(
        [("Ridge", reg_ridge_cv_r2.mean()), ("Random Forest", reg_rf_cv_r2.mean()), ("Gradient Boosting", reg_gb_cv_r2.mean())],
        key=lambda x: x[1],
    )
    mo.md(f"""
    **{_best_reg[0]}** achieves the best R-squared ({_best_reg[1]:.4f}), meaning features explain only
    about {_best_reg[1] * 100:.0f}% of popularity variance. The gap between Ridge (R-squared near zero) and
    tree-based models is dramatic — it proves there is no linear signal, and all predictive power comes from
    non-linear interactions captured by tree models. Even the best model leaves ~73% of variance unexplained,
    confirming that raw popularity is very difficult to predict from audio features and genre alone.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.7 Final Regression Model: Test Set Evaluation
    """)
    return


@app.cell
def _(
    alt,
    mean_squared_error,
    mo,
    np,
    pd,
    r2_score,
    reg_X_test,
    reg_X_train,
    reg_gb_cv_r2,
    reg_gb_pipe,
    reg_rf_cv_r2,
    reg_rf_pipe,
    reg_ridge_cv_r2,
    reg_ridge_pipe,
    reg_y_test,
    reg_y_train,
):
    _reg_cv_r2 = {
        "Ridge": reg_ridge_cv_r2.mean(),
        "Random Forest": reg_rf_cv_r2.mean(),
        "Gradient Boosting": reg_gb_cv_r2.mean(),
    }
    _best_reg_name = max(_reg_cv_r2, key=_reg_cv_r2.get)
    _reg_pipe_map = {"Ridge": reg_ridge_pipe, "Random Forest": reg_rf_pipe, "Gradient Boosting": reg_gb_pipe}
    reg_final_pipe = _reg_pipe_map[_best_reg_name]

    reg_final_pipe.fit(reg_X_train, reg_y_train)
    reg_final_pred = reg_final_pipe.predict(reg_X_test)

    reg_test_rmse = np.sqrt(mean_squared_error(reg_y_test, reg_final_pred))
    reg_test_r2 = r2_score(reg_y_test, reg_final_pred)

    _scatter_df = pd.DataFrame({"Actual": reg_y_test, "Predicted": reg_final_pred})

    reg_scatter_chart = (
        alt.Chart(pd.DataFrame(_scatter_df))
        .mark_circle(size=30, opacity=0.4)
        .encode(
            x=alt.X("Actual:Q", title="Actual Popularity"),
            y=alt.Y("Predicted:Q", title="Predicted Popularity"),
            tooltip=["Actual", alt.Tooltip("Predicted:Q", format=".1f")],
        )
        .properties(title=f"Actual vs Predicted — {_best_reg_name}", width=400, height=350)
    )

    _line = (
        alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]}))
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )

    mo.vstack([
        mo.md(f"**Final model:** {_best_reg_name} | **Test RMSE:** {reg_test_rmse:.2f} | **Test R-squared:** {reg_test_r2:.4f}"),
        reg_scatter_chart + _line,
    ])
    return reg_final_pipe, reg_final_pred


@app.cell
def _(mo):
    mo.md("""
    ### 4.8 Residual Analysis
    """)
    return


@app.cell
def _(alt, pd, reg_final_pred, reg_y_test):
    _residuals = reg_y_test - reg_final_pred
    _resid_df = pd.DataFrame({"Predicted": reg_final_pred, "Residual": _residuals})

    reg_residual_chart = (
        alt.Chart(pd.DataFrame(_resid_df))
        .mark_circle(size=30, opacity=0.4)
        .encode(
            x=alt.X("Predicted:Q", title="Predicted Popularity"),
            y=alt.Y("Residual:Q", title="Residual (Actual - Predicted)"),
            tooltip=["Predicted", alt.Tooltip("Residual:Q", format=".1f")],
        )
        .properties(title="Residual Plot", width=400, height=300)
    )

    _zero_line = (
        alt.Chart(pd.DataFrame({"x": [_resid_df["Predicted"].min(), _resid_df["Predicted"].max()], "y": [0, 0]}))
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )

    reg_residual_chart + _zero_line
    return


@app.cell
def _(mo):
    mo.md("""
    The residual plot shows whether prediction errors are systematic. Ideally, residuals should be
    randomly scattered around zero. Any patterns indicate the model is missing structure in the data.
    The model likely struggles at the extremes — underpredicting very popular tracks and overpredicting
    zero-popularity tracks.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.9 Feature Importance for Regression
    """)
    return


@app.cell
def _(alt, np, pl, reg_final_pipe):
    _reg_model = reg_final_pipe.named_steps["reg"]
    if hasattr(_reg_model, "feature_importances_"):
        _reg_importances = _reg_model.feature_importances_
    else:
        _reg_importances = np.abs(_reg_model.coef_)

    _reg_feat_names = reg_final_pipe.named_steps["preprocessor"].get_feature_names_out()
    _reg_feat_names = [f.split("__")[-1] for f in _reg_feat_names]

    _reg_imp_df = pl.DataFrame({"Feature": list(_reg_feat_names), "Importance": list(_reg_importances)}).sort("Importance", descending=True)

    reg_feat_imp_chart = (
        alt.Chart(_reg_imp_df.head(15))
        .mark_bar()
        .encode(
            x=alt.X("Importance:Q", title="Feature Importance"),
            y=alt.Y("Feature:N", sort="-x", title=""),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(title="Top 15 Feature Importances (Regression)", width=450, height=350)
    )
    reg_feat_imp_chart
    return


@app.cell
def _(mo):
    mo.md("""
    The feature importance ranking for regression is very similar to classification: acousticness, tempo,
    valence, and speechiness lead, with no single dominant feature. This consistency suggests the same
    features drive both tasks — the underlying signal is the same, and it is the task formulation
    (binary vs continuous) rather than different features that explains the performance gap.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.10 Classification vs Regression: Difficulty Comparison
    """)
    return


@app.cell
def _(
    clf_gb_cv,
    clf_lr_cv,
    clf_rf_cv,
    mo,
    reg_gb_cv_r2,
    reg_rf_cv_r2,
    reg_ridge_cv_r2,
):
    _best_clf_cv = max(clf_lr_cv.mean(), clf_rf_cv.mean(), clf_gb_cv.mean())
    _best_reg_cv_r2 = max(reg_ridge_cv_r2.mean(), reg_rf_cv_r2.mean(), reg_gb_cv_r2.mean())
    _ridge_r2 = reg_ridge_cv_r2.mean()

    mo.md(f"""
    Predicting the exact popularity score (regression) is **substantially harder** than predicting whether a
    track is above or below the median (classification).

    | Task | Best CV Metric | Linear Model |
    |------|---------------|--------------|
    | Classification | Accuracy: {_best_clf_cv:.4f} (RF) | LR: {clf_lr_cv.mean():.4f} |
    | Regression | R-squared: {_best_reg_cv_r2:.4f} (RF) | Ridge R²: {_ridge_r2:.4f} |

    **The Ridge R² of {_ridge_r2:.3f} is the key evidence**: it shows there is essentially NO linear signal
    between audio features and popularity. All predictive power comes from non-linear interactions
    captured by tree-based models.

    This difference is expected for several reasons:

    1. **Classification simplifies the problem** to a binary decision — the model only needs to distinguish
       "somewhat popular" from "somewhat unpopular". A track with popularity 40 vs 50 (near the median) is
       a hard call for regression but classification forgives the error as long as it falls on the correct side.

    2. **The popularity distribution has a large spike at zero** and a long tail. These extremes are hard to
       predict exactly. For classification, most zero-popularity tracks fall below the median, so they are
       correctly classified as "low" even if the model cannot predict their exact value.

    3. **Popularity is fundamentally driven by external factors** (artist fame, marketing budget, playlist
       placement, release timing) not captured in audio features. Audio features give a rough sense of
       commercial viability but cannot predict exact stream counts. The binary task is more forgiving because
       it only asks "is this track above average?" rather than "exactly how popular is it?"

    4. **Regression requires capturing the full range** (0-100), while classification collapses this to two bins.
       The information loss in binarisation actually helps — it removes noise that would hurt regression.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Conclusion
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Key Findings

    **EDA** revealed a dataset of 2000 Spotify tracks across 5 genres with a critical insight:
    **popularity has near-zero linear correlation with every audio feature** (all |r| ≤ 0.10).
    Energy and loudness correlate at 0.633, and genres differ in median popularity (pop: 66, synth-pop: 29),
    but no single feature predicts popularity linearly.

    **Clustering** produced weak clusters (all silhouette scores below 0.2). K-Means with k=5
    partially separates hip-hop and synth-pop into distinct clusters, but pop and indie-pop spread
    across all clusters. DBSCAN at eps=3.0 finds similar structure (silhouette ~0.19) but with loose
    neighbourhoods; at low eps, most points are noise. Genres are cultural constructs not fully captured
    by acoustic measurements.

    **Classification** achieved ~70% accuracy (Random Forest) in predicting above/below-median popularity,
    beating the ~60% of logistic regression. The 10-percentage-point gap between linear and tree-based models
    confirms the non-linear nature of the signal. No single feature dominates — all audio features contribute
    roughly equally (6-10% importance each).

    **Regression** confirmed that predicting exact popularity scores is substantially harder.
    Ridge R-squared of ~0.02 proves there is no linear signal at all. Random Forest achieves R-squared ~0.27,
    meaning features explain only about 27% of popularity variance with ~25-point RMSE on a 0-100 scale.

    **Overall**, this analysis demonstrates both the potential and the limitations of data mining on music data.
    The dominant finding is that audio features contain no linear signal for popularity — all predictive power
    comes from non-linear interactions captured by tree-based models. The most important drivers of track
    popularity — artist reputation, marketing, cultural trends — remain outside the dataset.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Appendix: Mistakes Found and Corrected

    During verification, we discovered several errors in the original notebook. Below is a complete list
    of every mistake, why it mattered, and how it was fixed.

    | # | Section | Original Mistake | Impact | Fix |
    |---|---------|-----------------|--------|-----|
    | 1 | 1.3 Data Cleaning | `explicit` column compared as `== "TRUE"` but the column is **Boolean**, not String | **Critical bug** — the comparison always returned False, zeroing out all 218 explicit=True values. Every downstream model trained on corrupted data. | Changed to `pl.col("explicit").cast(pl.Int32)` |
    | 2 | 1.1 Dataset Types | Described `explicit` as "stored as a string (`TRUE`/`FALSE`)" | Incorrect description of the data schema | Changed to "stored as a Boolean" |
    | 3 | 1.4 Popularity Distribution | No mention of popularity's near-zero correlation with audio features | Missed the single most important EDA finding — this shapes all modelling expectations | Added observation that popularity has near-zero linear correlation with all audio features (all abs(r) ≤ 0.10) |
    | 4 | 1.6 Correlation Analysis | Energy-loudness correlation reported as "~0.7" | Actual value is **0.633** — overstated by ~10% | Corrected to 0.633 |
    | 5 | 1.6 Correlation Analysis | No mention of popularity's weak correlations | Omitted a crucial finding for modelling | Added: "Popularity shows almost no linear correlation with any individual audio feature" |
    | 6 | 1.7 Genre Popularity | "indie-pop tends to be less popular" | **Wrong** — synth-pop (median 29) is the least popular, not indie-pop (median 47). R-n-b (42) was also omitted. | Corrected to list all 5 genres with their actual medians |
    | 7 | 1.9 EDA Summary | No mention of near-zero popularity correlations or implications for modelling | Failed to connect the most important EDA finding to downstream task expectations | Added the finding and noted: linear models expected to fail, ensemble models essential |
    | 8 | 2.3 Silhouette Analysis | Text implied scores varied meaningfully across k values | Scores are very flat (0.11–0.16), meaning no k produces well-separated clusters | Rewrote to emphasise the flat profile and its meaning |
    | 9 | 2.5 Genre Alignment | "Pop and indie-pop tracks are distributed across multiple clusters, while hip-hop tends to concentrate more" | Understated — hip-hop concentrates heavily in one cluster (144/292), synth-pop also concentrates (155/393) | Added specific concentration details for hip-hop and synth-pop |
    | 10 | 2.6 DBSCAN Code | Silhouette computed whenever `n_noise < len(labels)` (any non-trivial result) | When >50% of points are noise, silhouette of remaining tiny clusters is **artificially inflated** (0.99+), producing misleading metrics | Added filter: only compute silhouette when `noise_pct < 50%` |
    | 11 | 2.6 DBSCAN Commentary | "Everything collapses into a single cluster" at high eps | **Wrong** — at eps=3.0, DBSCAN finds 2-8 clusters with <7% noise and silhouette ~0.19, comparable to K-Means | Rewrote to accurately describe DBSCAN's behaviour across eps range |
    | 12 | 2.8 DBSCAN Genre | "Most tracks either fall into noise or are lumped into one dominant cluster" | Only true at low eps; at eps=3.0 DBSCAN finds multiple clusters with low noise | Rewrote to discuss eps-dependent behaviour |
    | 13 | 2.9 Clustering Conclusion | "DBSCAN failed entirely" | Overly dismissive — DBSCAN at eps=3.0 achieves silhouette ~0.19, comparable to K-Means ~0.16 | Rewrote with actual numbers and nuanced comparison |
    | 14 | 3.7 Classification Commentary | Generic statement about "modest accuracy" | Did not highlight the key finding: tree models (RF: 70.2%) significantly outperform linear (LR: 60.7%), confirming non-linearity | Added specific comparison and connected to EDA correlation finding |
    | 15 | 3.9 Feature Importance | "Genre indicators contribute to the prediction" implied genre was important | **Wrong** — genre features rank *lower* than all audio features. No single feature dominates (all 6-10%). | Corrected to describe the flat importance profile and genre's low ranking |
    | 16 | 4.3 Ridge Commentary | "Low R-squared" — vague | R² = 0.018 is not just "low", it's essentially **zero** — the most striking result in the entire regression analysis, proving no linear signal exists | Rewrote to emphasise the near-zero R² and its meaning |
    | 17 | 4.6 Regression Comparison | "All models show relatively poor R-squared values" | Missed the dramatic gap: Ridge R²=0.02 vs RF R²=0.27. This proves all signal is non-linear. | Added explicit comparison and interpretation of the gap |
    | 18 | 4.9 Regression Features | Generic "comparing with classification reveals whether same features drive both tasks" | Actually the features ARE very similar — this is a finding, not a question | Stated the finding directly: similar features, no dominant one |
    | 19 | 4.10 Comparison | No mention of Ridge R² as evidence | Ridge R²=0.018 is the single strongest piece of evidence for non-linearity | Added Ridge R² to comparison table and highlighted it as key evidence |
    | 20 | 5 Conclusion | Generic findings without specific numbers | Vague conclusions that could apply to any dataset | Rewrote with all actual numbers and the dominant theme: no linear signal |
    """)
    return


if __name__ == "__main__":
    app.run()
