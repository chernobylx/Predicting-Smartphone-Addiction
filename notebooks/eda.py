import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    from pathlib import Path
    from smartphone_addiction.paths import TRAIN_CSV, TEST_CSV, DATA_DIR
    from sklearn.model_selection import train_test_split as tts
    from scipy.stats import bootstrap as boot
    import numpy as np
    from functools import partial
    import altair as alt
    import seaborn as sns
    from matplotlib import pyplot as plt
    from sklearn.linear_model import LogisticRegressionCV as LgRCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import RocCurveDisplay
    from joblib import Memory
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import OneHotEncoder
    import miceforest as mf
    from sklearn.decomposition import PCA, NMF
    memory = Memory(location = './cache', verbose=0)
    return (
        DATA_DIR,
        LgRCV,
        NMF,
        PCA,
        TEST_CSV,
        TRAIN_CSV,
        alt,
        cs,
        memory,
        mf,
        mo,
        np,
        pl,
        plt,
        sns,
        tts,
    )


@app.cell
def _(pl):
    gender = pl.Categorical('gender')
    stress = pl.Enum([
        'Low',
        'Medium',
        'High'
    ])
    impact = pl.Categorical('work_impact')
    schema_override = pl.Schema(
        {
            'gender': gender,
            'stress_level': stress,
            'academic_work_impact': impact
        }
    )
    return (schema_override,)


@app.cell
def _(TEST_CSV, TRAIN_CSV, cs, pl, schema_override):
    train = pl.read_csv(TRAIN_CSV, schema_overrides=schema_override)
    test = pl.read_csv(TEST_CSV, schema_overrides = schema_override)
    og = pl.read_csv(r'data\raw\original.csv', schema_overrides={
        'stress_level': pl.Enum(['Low', 'Medium', 'High']),
        'gender': pl.Categorical('gender'),
        'academic_work_impact': pl.Categorical('impact'),
        'addiction_level': pl.Enum(['None', 'Mild', 'Moderate', 'Severe'])
    })
    target = cs.by_name('addicted_label')
    preds = cs.exclude(target, 'id')
    return og, preds, target, test, train


@app.cell
def _(og):
    og
    return


@app.cell
def _(train):
    train
    return


@app.cell
def _(test):
    test
    return


@app.cell
def _(np, target, train, tts):
    SEED = 42
    rng = np.random.default_rng(seed=SEED)
    eda,_ = tts(
        train,
        train_size = .1, 
        random_state = SEED, 
        stratify = train.select(target)
    )
    eda
    return SEED, eda


@app.cell
def _(eda, mo):
    mo.ui.dataframe(eda)
    return


@app.cell
def _(cs, eda):
    eda.describe().with_columns(cs.numeric().round(2))
    return


@app.cell
def _(eda, preds, target):
    eda.group_by(
        target
    ).agg(
        preds.is_null().mean()
    )
    return


@app.cell
def _(eda):
    eda.shape
    return


@app.cell
def _(DATA_DIR, eda, pl):
    _n = 8
    _i = 10
    n_boot = _i*2**_n
    for i in range(_i):
        f = DATA_DIR/'processed'/f'eda_boot_{i:02}.parquet'
        if not f.exists():
            pl.concat(
                eda.sample(
                    fraction = 1,
                    with_replacement = True,
                    seed = j,
                ).with_columns(
                    pl.lit(j).alias('sample')
                ) 
                for j in range(i*2**_n, (i+1)*2**_n)
            ).write_parquet(
                f,
                compression = 'zstd',
                compression_level = 1,
            )
    #eda_boot.to_parquet(DATA_DIR/raw/'eda_boot_1.parquet')
    return


@app.cell
def _(DATA_DIR, pl):
    B = pl.scan_parquet(DATA_DIR/'processed'/'eda_boot_*.parquet')
    return (B,)


@app.cell
def _(B, cs, pl, preds, target):
    null_ci = B.drop('id').group_by(
        target, 'sample'
    ).agg(
        preds.is_null().mean()
    ).drop('sample').group_by(
        target
    ).agg(
        preds.mean().name.suffix('_null_mean'),
        preds.quantile(.025).name.suffix('_null_ciLow'),
        preds.quantile(.975).name.suffix('_null_ciHigh'),
        #preds.sum().name.suffix('_null_count')
    ).unpivot(
        index = target
    ).with_columns(
        pl.col('variable').str.extract(r"^(.*)_null", 1).alias('column'),
        pl.col('variable').str.extract(r"([^_]*)$").alias('stat')
    ).drop(
        'variable'
    ).pivot(
        on = 'stat',
        on_columns = ['ciLow', 'ciHigh', 'mean'],
        values = 'value',
        index = cs.exclude('stat', 'value')
    )
    return (null_ci,)


@app.cell
def _(null_ci):
    N = null_ci.collect()
    return (N,)


@app.cell
def _(N, alt):
    base = alt.Chart(N).encode(
        x = alt.X('column:N', title = 'Predictor'),
        xOffset=alt.XOffset('addicted_label:N'),
        color = alt.Color('addicted_label:N'),
        tooltip=[
                "column",
                "addicted_label:N",
                alt.Tooltip("mean", format=".2%"),
                alt.Tooltip("ciLow", format=".2%"),
                alt.Tooltip("ciHigh", format=".2%"),
            ],
    )

    mean = base.mark_point(filled = True, size = 70).encode(
        y = alt.Y("mean:Q")
    )
    interval = base.mark_errorbar(ticks=True).encode(
        y = alt.Y('ciLow:Q', title = 'Null Rate',axis=alt.Axis(format="%")),
        y2 = 'ciHigh:Q'
    )
    (mean+interval).properties(
        width = 800,
        height = 400,
        title = alt.Title('Bootstrapped Confidence Interval (n = 10,000) of the Proportion of Null Values for Each Predictor by Target Label')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hypothesis
    It is not possible to predict the location of null values in a column better than random guessing using logistic regression.
    ## Procedure
    1. For each predictor column, remove the feature and then perform basic feature engineering and preprocessing on the remaining predictors.
    2. Train a logistic classifier to predict the presence of null values in the dropped predictor
    3. Compare the classifier to a random classifier
    4. Repeat for all samples in the bootstrap
    """)
    return


@app.cell
def _(LgRCV, cs, memory, np, pl):


    def pp(_df: pl.DataFrame):
            _df = _df.to_dummies(
                cs.categorical(),

                drop_nulls = True
            ).with_columns(
                cs.enum().to_physical()
            ).with_columns(
                (cs.numeric()-cs.numeric().mean()).truediv(cs.numeric().std()),
                pl.all().is_null().name.suffix('_null'),
            ).with_columns(
                pl.all().fill_null(strategy='mean')
            )
            return _df
    @memory.cache
    def fit_null_clf(df, col, scoring='roc_auc', penalty='elasticnet', l1_ratios=np.linspace(0,1,5), Cs=5, solver='saga', n_jobs=-1, verbose=0):


        y = df.select(pl.col(col).is_null())
        X = df.drop(col)

        X = pp(X)
        mod = LgRCV(scoring='roc_auc', penalty='elasticnet', l1_ratios=np.linspace(0,1,5), Cs=5, solver='saga', n_jobs=-1, verbose=0)
        mod.fit(X, y)

        return mod

 

    return fit_null_clf, pp


@app.cell
def _(B, fit_null_clf, np, pl, pp, preds):
    _n = 1
    _s = range(0,_n)#rng.integers(0,n_boot,_n)
    _v = 10
    s = _s[0]



    _val = B.filter(pl.col('sample').eq(_v)).drop('sample').select(preds).collect()

    results = {
        col : np.zeros(_n) for col in _val.schema.names()
    }
    for s in _s:
        for col in _val.schema.names():
            print(col, s)
            _y_val = _val.select(pl.col(col).is_null())
            _X_val = _val.drop(col)
            _X_val = pp(_X_val)
            _df = B.filter(pl.col('sample').eq(s)).drop('sample').select(preds).collect()

            results[col][s] = fit_null_clf(_df, col).score(_X_val, _y_val)
    #print(results)
    return (results,)


@app.cell
def _(pl, results):
    pl.DataFrame(results)
    return


@app.cell
def _(pl, sklearn):
    def get_df(trnsfrmr:sklearn.base.TransformerMixin, df:pl.DataFrame):
        return pl.DataFrame(trnsfrmr.transform(df), schema = trnsfrmr.get_feature_names_out().tolist())

    return


@app.cell
def _(SEED, cs, eda, mf, preds):

    _df = eda.select(
        preds
    )


    kernel = mf.ImputationKernel(
        _df.with_columns(
            cs.enum().to_physical()
        ).to_pandas(),
        num_datasets = 1,
        random_state = SEED
    )

    kernel.mice(5)
    print(kernel)
    return (kernel,)


@app.cell
def _(kernel):
    kernel.complete_data(0)
    return


@app.cell
def _(cs, eda, kernel, pl, plt, target):
    _eda = pl.concat([
        pl.DataFrame(kernel.complete_data(0)).to_dummies(cs.categorical(), drop_first=True),
        eda.select(target)
    ], how = 'horizontal')
    _eda = _eda.with_columns(
        (24-pl.col('sleep_hours')).alias('awake_hours'),
        (cs.matches('per_day').pow(-1)*24).name.replace(r'^(.*)_per_day$', "hours_per_${1}"),
        cs.matches('notification').truediv(cs.matches('app')).alias('notifications_per_app_open'),
        #cs.matches('daily').mul(7).sub(cs.matches('weekend')).truediv(5).alias('screen_time_hours_per_weekday'), #results in negative hours per weekday for certain individuals
        pl.col('daily_screen_time_hours').truediv(pl.col('weekend_screen_time')).alias('daily_screen_time_to_weekend_screen_time_ratio'),
        (cs.matches('_hours$')-cs.matches('sleep')).truediv(cs.matches('sleep')).name.suffix('_to_sleep_hours_ratio'),
   
    )
    _eda = _eda.with_columns(
        (cs.matches(r'_hours$')-cs.matches(r'^awake_hours$')).truediv(cs.matches(r'^awake_hours$')).name.suffix('_to_awake_hours_ratio')
    )

    _eda = _eda.with_columns(
        cs.matches('_ratio$').log1p().name.suffix('_log1p'),
        cs.matches('per').log1p().name.suffix('_log1p')
    
    )
    _eda =  _eda.with_columns(#be sure to drop the addicted_label first
        pl.mean_horizontal(pl.all()).alias('row_mean'),
        pl.max_horizontal(pl.all()).alias('row_max'),
        pl.sum_horizontal(pl.all()).alias('row_sum'),
    )
    eda_2 = _eda

    fig, ax = plt.subplots(figsize=(20, 14))
    #sns.heatmap(_eda.to_pandas().corr(), xticklabels=True, yticklabels=True, annot=True, ax = ax)
    #sns.pairplot(_eda.sample(100).to_pandas())
    #mo.ui.data_explorer(_eda.sample(1000, seed = SEED))
    _eda
    return (eda_2,)


@app.cell
def _(B, cs, pl, preds, target):
    from itertools import combinations
    _df = B.filter(pl.col('sample')<2**8).drop('id').collect().to_dummies(
        'gender',
        drop_nulls=True
    ).with_columns(
        cs.enum().to_physical()
    )

    _cols = cs.expand_selector(_df.drop(target, 'sample'),cs.numeric())
    _df = _df.group_by(
        target, 'sample'
    ).agg(
        [
            pl.corr(c1,c2).alias(f'{c1}_with_{c2}')
            for c1, c2 in combinations(_cols, 2)
        ]    
    )

    _df.drop('sample').group_by(
        target
    ).agg(
        preds.mean().name.suffix('_corr_mean'),
        preds.quantile(.025).name.suffix('_corr_ciLow'),
        preds.quantile(.975).name.suffix('_corr_ciHigh'),
        #preds.sum().name.suffix('_null_count')
    ).unpivot(
        index = target
    ).with_columns(
        pl.col('variable').str.extract(r"^(.*)_corr", 1).alias('column'),
        pl.col('variable').str.extract(r"([^_]*)$").alias('stat')
    ).drop(
        'variable'
    ).pivot(
        on = 'stat',
        on_columns = ['ciLow', 'ciHigh', 'mean'],
        values = 'value',
        index = cs.exclude('stat', 'value')
    ).pivot(
        on = target,
        index = 'column'
    ).filter(
        (pl.col('ciLow_1') > pl.col('ciHigh_0')) | (pl.col('ciLow_0') > pl.col('ciHigh_1'))
    )
    return


@app.cell
def _(eda, sns):
    sns.pairplot(eda.sample(500).to_pandas())
    return


@app.cell
def _(mo):
    mani = mo.ui.dropdown(['iso','lle', 'mlle','hessian', 'spectral', 'ltsa','tsne', 'umap'], value='iso')
    mani
    return (mani,)


@app.cell
def _(mo):
    dims = mo.ui.slider(2,5,1)
    dims
    return (dims,)


@app.cell
def _(mo):
    sample = mo.ui.slider(0,1,.01, .05)
    sample
    return (sample,)


@app.cell
def _(mo):
    neighbors = mo.ui.slider(1,50,1)
    neighbors
    return (neighbors,)


@app.cell
def _(mo):
    min_d = mo.ui.slider(0,.99,.01, .1)
    min_d
    return (min_d,)


@app.cell
def _(cs, eda, mo, pl, sample, target):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    _df = eda.drop_nulls().drop('id').sample(fraction = sample.value).to_dummies(cs.categorical()).with_columns(cs.enum().to_physical())
    X = _df.drop(target)
    y = _df.select(target)

    _X = StandardScaler().fit_transform(X)
    LOF = LocalOutlierFactor()
    SVM = OneClassSVM(gamma='auto')
    IF = IsolationForest(contamination=.1)
    ol = LOF.fit_predict(_X)
    svm = SVM.fit_predict(_X)
    #ifo = IF.fit_predict(_X)
    _d =pl.concat([
        pl.DataFrame(_X,schema = X.schema.names()),
        pl.DataFrame(svm, schema={'svm':int}),
        pl.DataFrame(ol, schema = {'ol':int}),
        #pl.DataFrame(ifo,schema={'ifo':int})
    ], how='horizontal')

    X = _d.filter(
        pl.col('ol')==1
    ).drop('ol','svm')#,'ifo')

    mo.ui.data_explorer(_d)
    return StandardScaler, X, y


@app.cell
def _(X, dims, mani, min_d, mo, neighbors, pl, y):
    from sklearn.manifold import Isomap
    from sklearn.manifold import LocallyLinearEmbedding as LLE
    from sklearn.manifold import SpectralEmbedding
    from sklearn.manifold import TSNE
    import umap

    n = dims.value
    m = mani.value
    if mani.value == 'iso':

        _mf = Isomap(n_components=dims.value)
    elif mani.value == 'lle':
        _mf = LLE(n_components=dims.value)
    elif mani.value == 'mlle':
        _mf = LLE(n_components = dims.value, method='modified', eigen_solver='dense', n_jobs=-1)
    elif m == 'hessian':
        _mf = LLE(n_components=n, method='hessian', n_neighbors=int(n*(n+3)), eigen_solver='dense', n_jobs=-1)
    elif m == 'spectral':
        _mf = SpectralEmbedding(n_components=n, n_jobs=-1)
    elif m == 'ltsa':
        _mf = LLE(n_components=n, method='ltsa', n_jobs=-1, eigen_solver='dense')
    elif m == 'tsne':
        _mf = TSNE(n_components=n, n_jobs=-1, perplexity=50)
    elif m == 'umap':
        _mf = umap.UMAP(n_components = n, n_neighbors = neighbors.value, min_dist=min_d.value)
    _df = pl.concat([pl.DataFrame(_mf.fit_transform(X)), y], how = 'horizontal')


    mo.ui.data_explorer(_df)
    return


@app.cell
def _(LgRCV, SEED, StandardScaler, eda_2, np, target, tts):
    _ss = StandardScaler()
    _X = eda_2.drop(target)
    _y = eda_2.select(target)
    _X_train, _X_val, _y_train, _y_val = tts(_X,_y, test_size = .1)
    _X_train = _ss.fit_transform(_X_train)
    _X_val = _ss.transform(_X_val)
    _mod = LgRCV(Cs = 10, cv = 5, n_jobs=-1, l1_ratios=np.linspace(0,1,11), scoring='roc_auc', random_state=SEED, solver = 'saga', verbose = True)
    _mod.fit(_X_train, _y_train)
    _mod.score(_X_val, _y_val)
    return


@app.cell
def _(PCA, StandardScaler, eda_2, mo, pl, target):

    _df = PCA(n_components=3).fit_transform(StandardScaler().fit_transform(eda_2.drop(target)))
    _df = pl.concat([pl.DataFrame(_df), eda_2.select(target)], how='horizontal')
    mo.ui.data_explorer(_df)
    return


@app.cell
def _(NMF, eda_2, mo, pl, target):
    from sklearn.preprocessing import MinMaxScaler
    _df = NMF(n_components=4).fit_transform(MinMaxScaler().fit_transform(eda_2.drop(target)))
    _df = pl.concat([pl.DataFrame(_df), eda_2.select(target)], how='horizontal')
    mo.ui.data_explorer(_df)
    return


@app.cell
def _(cs, og):
    from flaml import AutoML
    settings = {
        "time_budget": 16,                      # Maximum time in seconds for training
        "metric": 'log_loss',# Optimization metric
        'seed':45,
        'eval_method': 'cv',
        "task": 'multiclass',               # Type of ML task
        "estimator_list": ['xgboost'] ,          # Use only XGBoost
        "verbose": 1,                           # Minimize output messages
        #'gpu_per_trial':2
    
    }
    _X = og.drop(['transaction_id', 'user_id', 'addiction_level', 'addicted_label']).with_columns(cs.enum().to_physical())
    _y = og.select(cs.matches('addict').to_physical())
    _X_train = _X.to_pandas()
    _y_train = _y.to_pandas()['addiction_level']
    _X_train
    automl = AutoML()
    automl.fit(X_train=_X_train, y_train=_y_train, **settings)
    print(automl.score(_X_train, _y_train))

    from sklearn.metrics import confusion_matrix as cf
    cf(automl.predict(_X_train)>=2, _y_train>=2)
    return (automl,)


@app.cell
def _(automl):
    automl
    return


if __name__ == "__main__":
    app.run()
