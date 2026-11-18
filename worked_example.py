"""
Worked example: cleaning, feature engineering, outlier treatment and model
comparison for a classification task (customer churn) and a regression task
(house prices). All data is synthetic and deliberately messy so every step has
a visible purpose.

Run:  python worked_example.py
Needs: numpy, pandas, scikit-learn (>=1.3)
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------------
# Reusable transformer: winsorise (cap) numeric columns at percentiles that are
# LEARNED FROM TRAINING DATA ONLY (fit) and applied everywhere (transform).
# ----------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


# ============================================================================
# PART A - CLASSIFICATION: predict customer churn
# ============================================================================
def make_churn_data(n=15000):
    age = RNG.normal(42, 13, n).clip(18, 85).round()
    tenure = RNG.integers(1, 73, n).astype(float)
    contract = RNG.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    region = RNG.choice(["Accra", "Kumasi", "Takoradi", "Tamale"], n, p=[.4, .3, .2, .1])
    monthly = (30 + 0.6 * age + RNG.normal(0, 12, n)).clip(10, None).round(2)
    income = np.exp(RNG.normal(8.2, 0.7, n)).round()            # right-skewed, heavy tail
    support_calls = RNG.poisson(1.5, n).astype(float)
    last_login_days = RNG.exponential(12, n).round()
    signup = pd.Timestamp("2024-01-01") - pd.to_timedelta(tenure * 30, unit="D")

    # True churn process (non-linear on purpose: an interaction + threshold effects)
    z = (-2.6
         + 1.6 * (contract == "Month-to-month")
         - 1.4 * (contract == "Two year")
         + 0.40 * support_calls
         + 0.05 * last_login_days
         - 0.03 * tenure
         + 1.3 * ((monthly > 70) & (contract == "Month-to-month"))   # interaction
         + 1.1 * ((support_calls >= 4) & (tenure < 12))               # interaction
         - 0.5 * (np.log(income) - 8.2))                              # richer customers churn less
    churn = (RNG.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame(dict(customer_id=np.arange(n), age=age, tenure_months=tenure,
                           contract=contract, region=region, monthly_charges=monthly,
                           income=income, support_calls=support_calls,
                           last_login_days=last_login_days, signup_date=signup, churn=churn))
    # Leaky column: only exists AFTER a customer has churned
    df["cancellation_reason"] = np.where(df.churn == 1,
                                         RNG.choice(["price", "service", "moved"], n), None)

    # ---- inject real-world mess ----
    df.loc[RNG.random(n) < 0.12, "income"] = np.nan                       # missing income
    older = df.age > 55
    df.loc[older & (RNG.random(n) < 0.15), "income"] = np.nan             # MAR: older skip it
    df.loc[RNG.random(n) < 0.03, "tenure_months"] = np.nan
    bad = RNG.random(n) < 0.015
    df.loc[bad, "age"] = RNG.choice([-5, 0, 150, 999], bad.sum())          # impossible ages
    df.loc[RNG.random(n) < 0.01, "monthly_charges"] = 9999.0              # sentinel value
    df.loc[RNG.random(n) < 0.02, "income"] = df.income.max() * 8          # genuine extreme earners
    df["contract"] = [c if r > 0.3 else RNG.choice([c.lower(), c.upper(), c + " "])
                      for c, r in zip(df.contract, RNG.random(n))]         # messy labels
    df["region"] = [r if x > 0.1 else r.lower() for r, x in zip(df.region, RNG.random(n))]
    df = pd.concat([df, df.sample(300, random_state=1)], ignore_index=True)  # duplicate rows
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


def clean_churn(df, verbose=True):
    """Deterministic cleaning: rules that do not learn from data, so safe before splitting."""
    log = {"rows_before": len(df)}
    df = df.drop_duplicates()                                    # duplicates would leak across splits
    log["rows_after_dedup"] = len(df)
    df["contract"] = df.contract.str.strip().str.title()          # 'MONTH-TO-MONTH ' -> 'Month-To-Month'
    df["region"] = df.region.str.strip().str.title()
    log["bad_age"] = int(((df.age < 18) | (df.age > 100)).sum())
    df.loc[(df.age < 18) | (df.age > 100), "age"] = np.nan        # impossible -> missing (imputed later)
    log["sentinel_charges"] = int((df.monthly_charges > 1000).sum())
    df.loc[df.monthly_charges > 1000, "monthly_charges"] = np.nan  # 9999 is a placeholder, not a price
    df = df.drop(columns=["cancellation_reason"])                 # LEAKAGE: known only after churn
    if verbose:
        print("cleaning log:", log)
    return df.drop_duplicates()


def engineer_churn(df):
    """Feature engineering that uses only information available at prediction time."""
    df = df.copy()
    df["income_missing"] = df.income.isna().astype(int)           # missingness itself can be signal
    df["charges_per_tenure"] = df.monthly_charges / df.tenure_months.clip(lower=1)
    df["calls_per_year"] = df.support_calls / (df.tenure_months.clip(lower=1) / 12)
    df["new_customer"] = (df.tenure_months < 12).astype(int)
    df["signup_month"] = df.signup_date.dt.month
    df["log_income"] = np.log1p(df.income)                        # tame right skew
    return df.drop(columns=["customer_id", "signup_date", "income"])


def churn_preprocessor(num_cols, cat_cols, outlier_treatment=True, scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if outlier_treatment:
        steps.append(("winsorize", Winsorizer(0.01, 0.99)))
    if scale:
        steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", Pipeline(steps), num_cols), ("cat", cat, cat_cols)])


def run_classification():
    print("\n" + "=" * 78 + "\nPART A - CLASSIFICATION (customer churn)\n" + "=" * 78)
    raw = make_churn_data()
    print(f"raw shape {raw.shape} | churn rate {raw.churn.mean():.1%} | "
          f"missing income {raw.income.isna().mean():.1%}")

    # ---------- Scenario 1: 'naive' - no cleaning, leak left in -------------
    naive = raw.drop(columns=["customer_id", "signup_date"]).copy()
    naive["cancellation_reason"] = naive.cancellation_reason.fillna("none")
    Xn, yn = naive.drop(columns="churn"), naive.churn
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(Xn, yn, test_size=0.2, stratify=yn, random_state=42)
    ncat = Xn.select_dtypes("object").columns.tolist()
    nnum = [c for c in Xn.columns if c not in ncat]
    leaky = Pipeline([("prep", churn_preprocessor(nnum, ncat, outlier_treatment=False)),
                      ("m", HistGradientBoostingClassifier(random_state=0))]).fit(Xn_tr, yn_tr)
    print("\n[Naive pipeline WITH leaked column] test ROC-AUC:",
          round(roc_auc_score(yn_te, leaky.predict_proba(Xn_te)[:, 1]), 3),
          "  <-- suspiciously perfect")

    # ---------- Scenario 2: cleaned + engineered ------------------------------
    df = engineer_churn(clean_churn(raw))
    X, y = df.drop(columns="churn"), df.churn
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cat_cols = X.select_dtypes("object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"clean shape {df.shape}; train {len(X_tr)}, test {len(X_te)}")

    candidates = {
        "Baseline (majority class)": Pipeline([("p", churn_preprocessor(num_cols, cat_cols)),
                                               ("m", DummyClassifier(strategy="prior"))]),
        "Logistic Regression": Pipeline([("p", churn_preprocessor(num_cols, cat_cols)),
                                         ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("p", churn_preprocessor(num_cols, cat_cols, scale=False)),
                                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                class_weight="balanced_subsample",
                                                                n_jobs=-1, random_state=0))]),
        "Gradient Boosting": Pipeline([("p", churn_preprocessor(num_cols, cat_cols, scale=False)),
                                       ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                             max_leaf_nodes=15, l2_regularization=1.0,
                                                                             random_state=0))]),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, pipe in candidates.items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        cv_ap = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="average_precision").mean()
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        rows.append(dict(model=name, cv_roc_auc=cv_auc, cv_pr_auc=cv_ap,
                         test_roc_auc=roc_auc_score(y_te, p), test_pr_auc=average_precision_score(y_te, p)))
    res = pd.DataFrame(rows).set_index("model").round(3)
    print("\nCleaned + engineered features, 5-fold CV on train; final numbers on untouched test set")
    print(res.to_string())

    # ---------- Ablation: effect of each step on the best simple + best flexible model
    def score_variant(label, frame, model_fn, outlier=True, scale=True):
        Xv, yv = frame.drop(columns="churn"), frame.churn
        Xa, Xb, ya, yb = train_test_split(Xv, yv, test_size=0.2, stratify=yv, random_state=42)
        cc = Xv.select_dtypes("object").columns.tolist()
        nc = [c for c in Xv.columns if c not in cc]
        pipe = Pipeline([("p", churn_preprocessor(nc, cc, outlier, scale)), ("m", model_fn())]).fit(Xa, ya)
        return round(roc_auc_score(yb, pipe.predict_proba(Xb)[:, 1]), 3)

    raw_noleak = raw.drop(columns=["customer_id", "signup_date", "cancellation_reason"])
    base_clean = clean_churn(raw, verbose=False)
    base_noeng = base_clean.drop(columns=["customer_id", "signup_date"])
    lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
    gb = lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                l2_regularization=1.0, random_state=0)
    abl = pd.DataFrame({
        "Logistic Regression": [score_variant("", raw_noleak, lr, outlier=False),
                                score_variant("", base_noeng, lr, outlier=False),
                                score_variant("", base_noeng, lr, outlier=True),
                                score_variant("", df, lr, outlier=True)],
        "Gradient Boosting": [score_variant("", raw_noleak, gb, outlier=False, scale=False),
                              score_variant("", base_noeng, gb, outlier=False, scale=False),
                              score_variant("", base_noeng, gb, outlier=True, scale=False),
                              score_variant("", df, gb, outlier=True, scale=False)],
    }, index=["raw data, no cleaning (leak column already dropped)",
              "+ cleaning (dedupe, labels, impossible/sentinel values)",
              "+ outlier capping (winsorise 1-99%)",
              "+ feature engineering"])
    print("\nAblation - test ROC-AUC as each step is added:")
    print(abl.to_string())

    # ---------- Threshold / business view for the best model ------------------
    best = candidates["Gradient Boosting"]
    p = best.predict_proba(X_te)[:, 1]
    print("\nGradient Boosting - precision/recall trade-off by threshold (test set)")
    for t in (0.5, 0.35, 0.25):
        pred = (p >= t).astype(int)
        print(f"  threshold {t:.2f}: precision {precision_score(y_te, pred):.2f} | "
              f"recall {recall_score(y_te, pred):.2f} | F1 {f1_score(y_te, pred):.2f}")
    print(f"  (always predicting 'no churn' would be {1 - y_te.mean():.1%} accurate but catch 0% of churners)")
    return res, abl


# ============================================================================
# PART B - REGRESSION: predict house price
# ============================================================================
def make_house_data(n=4000):
    sqft = RNG.lognormal(7.3, 0.35, n).round()
    bedrooms = np.clip((sqft / 600 + RNG.normal(0, 0.7, n)).round(), 1, 8)
    age = RNG.integers(0, 60, n).astype(float)
    hood = RNG.choice(["Airport", "Osu", "Tema", "Madina", "Labone"], n, p=[.25, .15, .25, .2, .15])
    hood_mult = pd.Series(hood).map({"Airport": 1.6, "Osu": 1.8, "Tema": 0.9, "Madina": 0.8, "Labone": 1.5}).values
    lot = (sqft * RNG.uniform(1.2, 3.5, n)).round()
    price = (sqft ** 1.1) * 38 * hood_mult * np.exp(-0.006 * age) * np.exp(RNG.normal(0, 0.18, n))
    df = pd.DataFrame(dict(sqft=sqft, bedrooms=bedrooms, age=age, neighborhood=hood, lot_size=lot, price=price.round()))
    # genuine luxury outliers (real, keep info) and data-entry errors (fix)
    lux = RNG.random(n) < 0.01
    df.loc[lux, "price"] *= RNG.uniform(4, 9, lux.sum())
    typo = RNG.random(n) < 0.012
    df.loc[typo, "sqft"] *= 10                                           # extra zero typed
    df.loc[RNG.random(n) < 0.01, "sqft"] = 15                            # placeholder / unit error
    df.loc[RNG.random(n) < 0.05, "lot_size"] = np.nan
    df.loc[RNG.random(n) < 0.03, "age"] = np.nan
    df["neighborhood"] = [h if r > 0.1 else h.upper() for h, r in zip(df.neighborhood, RNG.random(n))]
    return df


def run_regression():
    print("\n" + "=" * 78 + "\nPART B - REGRESSION (house prices)\n" + "=" * 78)
    raw = make_house_data()
    print(f"raw shape {raw.shape} | price median {raw.price.median():,.0f} | max {raw.price.max():,.0f} "
          f"| sqft range {raw.sqft.min():.0f} - {raw.sqft.max():.0f}")

    df = raw.copy()
    df["neighborhood"] = df.neighborhood.str.title()
    n_bad = int(((df.sqft < 200) | (df.sqft > 15000)).sum())
    df.loc[(df.sqft < 200) | (df.sqft > 15000), "sqft"] = np.nan         # data-entry errors -> missing
    print(f"data-entry errors in sqft set to missing: {n_bad}")
    df["rooms_per_1000sqft"] = df.bedrooms / (df.sqft / 1000)             # engineered ratio
    df["lot_to_house"] = df.lot_size / df.sqft

    X, y = df.drop(columns="price"), df.price
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    cat = ["neighborhood"]
    num = [c for c in X.columns if c not in cat]

    def prep(outlier=True, scale=True):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if outlier:
            steps.append(("winsorize", Winsorizer(0.01, 0.99)))
        if scale:
            steps.append(("scale", StandardScaler()))
        return ColumnTransformer([("num", Pipeline(steps), num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

    log_t = dict(func=np.log1p, inverse_func=np.expm1)
    models = {
        "Baseline (mean price)": Pipeline([("p", prep()), ("m", DummyRegressor())]),
        "Linear Regression - raw target, no outlier cap": Pipeline([("p", prep(outlier=False)), ("m", LinearRegression())]),
        "Linear Regression - + outlier cap": Pipeline([("p", prep()), ("m", LinearRegression())]),
        "Ridge - + outlier cap + log target": TransformedTargetRegressor(
            Pipeline([("p", prep()), ("m", Ridge(alpha=1.0))]), **log_t),
        "Random Forest (raw target)": Pipeline([("p", prep(outlier=False, scale=False)),
                                                ("m", RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0))]),
        "Gradient Boosting (raw target)": Pipeline([("p", prep(outlier=False, scale=False)),
                                                    ("m", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, random_state=0))]),
        "Gradient Boosting - log target": TransformedTargetRegressor(
            Pipeline([("p", prep(outlier=False, scale=False)),
                      ("m", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, random_state=0))]), **log_t),
    }
    # no-cleaning comparison: same split, but typos/placeholders in sqft left in place
    raw_feat = raw.copy()
    raw_feat["neighborhood"] = raw_feat.neighborhood.str.title()
    raw_feat["rooms_per_1000sqft"] = raw_feat.bedrooms / (raw_feat.sqft / 1000)
    raw_feat["lot_to_house"] = raw_feat.lot_size / raw_feat.sqft
    Xr = raw_feat.drop(columns="price")
    Xr_tr, Xr_te = Xr.loc[X_tr.index], Xr.loc[X_te.index]
    uncleaned = {
        "Linear Regression - NO cleaning of sqft errors": Pipeline([("p", prep(outlier=False)), ("m", LinearRegression())]),
        "Random Forest - NO cleaning of sqft errors": Pipeline([("p", prep(outlier=False, scale=False)),
                                                                ("m", RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0))]),
    }
    rows = []
    for name, m in uncleaned.items():
        m.fit(Xr_tr, y_tr)
        p = m.predict(Xr_te)
        rows.append(dict(model=name, MAE=mean_absolute_error(y_te, p),
                         RMSE=mean_squared_error(y_te, p) ** 0.5, R2=r2_score(y_te, p),
                         median_abs_pct_err=float(np.median(np.abs(p - y_te) / y_te) * 100)))
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        rows.append(dict(model=name, MAE=mean_absolute_error(y_te, p),
                         RMSE=mean_squared_error(y_te, p) ** 0.5, R2=r2_score(y_te, p),
                         median_abs_pct_err=float(np.median(np.abs(p - y_te) / y_te) * 100)))
    res = pd.DataFrame(rows).set_index("model")
    print("\nTest-set results (original currency units):")
    print(res.round({"MAE": 0, "RMSE": 0, "R2": 3, "median_abs_pct_err": 1}).to_string())
    return res


if __name__ == "__main__":
    run_classification()
    run_regression()
