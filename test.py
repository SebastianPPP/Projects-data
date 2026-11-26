import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)


class GroupSeparateLinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, group_col='group', alpha=1e-6):
        self.group_col = group_col
        self.alpha = alpha

    def fit(self, X, y):
        self.models_ = {}
        self.feature_cols_ = [c for c in X.columns if c != self.group_col]
        y_arr = np.asarray(y)
        for grp in X[self.group_col].unique():
            mask = (X[self.group_col] == grp).values
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('reg', Ridge(alpha=self.alpha)),
            ])
            pipe.fit(X.loc[mask, self.feature_cols_], y_arr[mask])
            self.models_[grp] = pipe
        return self

    def predict(self, X):
        preds = np.zeros(len(X), dtype=float)
        for grp, pipe in self.models_.items():
            mask = (X[self.group_col] == grp).values
            if np.any(mask):
                preds[mask] = pipe.predict(X.loc[mask, self.feature_cols_])
        return preds


output_dir = os.path.join('images', 'project1')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('data.csv')

sns.set_theme(style='whitegrid', context='notebook')
palette = {'group1': '#1f77b4', 'group2': '#ff7f0e'}

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=100)
plt.subplots_adjust(hspace=0.38, wspace=0.28)

ax = axes[0, 0]
np.random.seed(42)
for grp, sub in df.groupby('group'):
    jitter = np.random.uniform(-0.12, 0.12, size=len(sub))
    ax.scatter(
        sub['time'] + jitter,
        sub['dependent'],
        alpha=0.45,
        s=18,
        color=palette[grp],
        label=f'{grp} points',
    )
    means = sub.groupby('time')['dependent'].mean()
    ax.plot(
        means.index,
        means.values,
        color=palette[grp],
        linewidth=2.2,
        marker='o',
        markersize=4,
        label=f'{grp} mean',
    )
ax.set_title('1. Dependent vs. Time', fontsize=10, fontweight='bold')
ax.set_xlabel('time', fontsize=9)
ax.set_ylabel('dependent', fontsize=9)
ax.set_xticks(range(1, 11))
ax.legend(fontsize=7.5, loc='upper left')

ax = axes[0, 1]
corr_g1 = df[df['group'] == 'group1'].drop(columns=['group']).corr()['dependent'].drop('dependent')
corr_g2 = df[df['group'] == 'group2'].drop(columns=['group']).corr()['dependent'].drop('dependent')
corr_df = pd.DataFrame({'group1': corr_g1, 'group2': corr_g2})
sns.heatmap(
    corr_df,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={'label': 'Pearson r'},
    ax=ax,
    linewidths=1,
    annot_kws={'size': 8.5},
)
ax.set_title('2. Pearson Correlation by Group', fontsize=10, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)

vars_to_plot = ['var1', 'var2', 'var3', 'var4']
feature_axes = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
titles = [
    '3. Dependent vs. var1',
    '4. Dependent vs. var2',
    '5. Dependent vs. var3',
    '6. Dependent vs. var4',
]

for ax, var, title in zip(feature_axes, vars_to_plot, titles):
    for grp, sub in df.groupby('group'):
        ax.scatter(sub[var], sub['dependent'], alpha=0.45, s=18, color=palette[grp], label=grp)
        m, b = np.polyfit(sub[var], sub['dependent'], 1)
        x_vals = np.linspace(sub[var].min(), sub[var].max(), 100)
        ax.plot(x_vals, m * x_vals + b, color=palette[grp], linewidth=1.8, linestyle='--')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(var, fontsize=9)
    ax.set_ylabel('dependent', fontsize=9)
    ax.legend(fontsize=8)

fig.suptitle('Exploratory Data Analysis: Group Regimes & Predictors (n = 400)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_overview.png'), dpi=150, bbox_inches='tight')
plt.show()

df_res = df.copy()
for grp in ['group1', 'group2']:
    mask = df_res['group'] == grp
    time_model = smf.ols('dependent ~ time + I(time**2) + I(time**3)', data=df_res[mask]).fit()
    df_res.loc[mask, 'dep_detrended'] = time_model.resid

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), dpi=100)
for ax, var in zip(axes, vars_to_plot):
    for grp, sub in df_res.groupby('group'):
        ax.scatter(sub[var], sub['dep_detrended'], alpha=0.5, s=18, color=palette[grp], label=grp)
        m, b = np.polyfit(sub[var], sub['dep_detrended'], 1)
        x_vals = np.linspace(sub[var].min(), sub[var].max(), 100)
        ax.plot(x_vals, m * x_vals + b, color=palette[grp], linewidth=1.8, linestyle='--')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_title(f'Detrended vs. {var}', fontsize=9.5, fontweight='bold')
    ax.set_xlabel(var, fontsize=8.5)
    if var == 'var1':
        ax.set_ylabel('Time-Detrended Residual', fontsize=8.5)
    ax.legend(fontsize=7.5)

fig.suptitle('Partial Predictor Effects After Removing Group Time Trends', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_detrended.png'), dpi=150, bbox_inches='tight')
plt.show()

X = df[['group', 'time', 'var1', 'var2', 'var3', 'var4']]
y = df['dependent'].values

sep_model_full = GroupSeparateLinearRegressor(group_col='group', alpha=1e-6).fit(X, y)
time_grid = np.linspace(df['time'].min(), df['time'].max(), 100)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=100)
for idx, grp in enumerate(['group1', 'group2']):
    ax = axes[idx]
    sub = df[df['group'] == grp]
    ax.scatter(sub['time'], sub['dependent'], alpha=0.5, s=20, color=palette[grp], label=f'{grp} data')
    grid_df = pd.DataFrame({
        'group': grp,
        'time': time_grid,
        'var1': sub['var1'].mean(),
        'var2': sub['var2'].mean(),
        'var3': sub['var3'].mean(),
        'var4': sub['var4'].mean(),
    })
    y_line = sep_model_full.predict(grid_df)
    slope = (y_line[-1] - y_line[0]) / (time_grid[-1] - time_grid[0])
    intercept = y_line[0] - slope * time_grid[0]
    ax.plot(
        time_grid,
        y_line,
        color=palette[grp],
        linewidth=2.6,
        label=f'y = {slope:.2f}·time + {intercept:.2f}',
    )
    ax.set_title(f'Separate Linear Model: {grp}', fontsize=10, fontweight='bold')
    ax.set_xlabel('time', fontsize=9)
    ax.set_ylabel('dependent', fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8.5, loc='upper left')

ax_both = axes[2]
for grp in ['group1', 'group2']:
    sub = df[df['group'] == grp]
    ax_both.scatter(sub['time'], sub['dependent'], alpha=0.35, s=16, color=palette[grp])
    grid_df = pd.DataFrame({
        'group': grp,
        'time': time_grid,
        'var1': sub['var1'].mean(),
        'var2': sub['var2'].mean(),
        'var3': sub['var3'].mean(),
        'var4': sub['var4'].mean(),
    })
    y_line = sep_model_full.predict(grid_df)
    slope = (y_line[-1] - y_line[0]) / (time_grid[-1] - time_grid[0])
    intercept = y_line[0] - slope * time_grid[0]
    ax_both.plot(
        time_grid,
        y_line,
        color=palette[grp],
        linewidth=2.6,
        label=f'{grp}: {slope:.2f}·t + {intercept:.2f}',
    )
ax_both.set_title('Both Separate Lines Compared', fontsize=10, fontweight='bold')
ax_both.set_xlabel('time', fontsize=9)
ax_both.set_ylabel('dependent', fontsize=9)
ax_both.set_xticks(range(1, 11))
ax_both.legend(fontsize=8, loc='upper left')

fig.suptitle('Group-Separate Linear Regression: Two Independent Lines', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'group_separate_lines.png'), dpi=150, bbox_inches='tight')
plt.show()

num_cols = ['time', 'var1', 'var2', 'var3', 'var4']
cat_cols = ['group']

prep_linear = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols),
])

prep_spline = ColumnTransformer([
    ('time_spline', SplineTransformer(n_knots=5, degree=3, include_bias=False), ['time']),
    ('other_num', StandardScaler(), ['var1', 'var2', 'var3', 'var4']),
    ('cat', OneHotEncoder(drop='first'), cat_cols),
])

prep_tree = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols),
])

gb_default_pipe = Pipeline([
    ('prep', prep_tree),
    ('model', GradientBoostingRegressor(random_state=42)),
])

gb_param_dist = {
    'model__n_estimators': [100, 150, 200, 300, 400, 500],
    'model__learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    'model__max_depth': [2, 3, 4],
    'model__min_samples_leaf': [2, 4, 6, 10, 15],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__subsample': [0.6, 0.75, 0.85, 1.0],
    'model__max_features': [None, 'sqrt', 0.7, 0.85],
    'model__loss': ['squared_error', 'huber'],
}

inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
gb_tuned = RandomizedSearchCV(
    estimator=clone(gb_default_pipe),
    param_distributions=gb_param_dist,
    n_iter=60,
    scoring='neg_root_mean_squared_error',
    cv=inner_cv,
    n_jobs=-1,
    random_state=42,
)

models = {
    '1. Global Linear (OLS)': Pipeline([
        ('prep', prep_linear),
        ('model', Ridge(alpha=1e-6)),
    ]),
    '2. Group-Separate Linear': GroupSeparateLinearRegressor(group_col='group', alpha=1e-6),
    '3. Spline + Interactions': Pipeline([
        ('prep', prep_spline),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', Ridge(alpha=10.0)),
    ]),
    '4. Gradient Boosting (Default)': gb_default_pipe,
    '5. Gradient Boosting (Tuned)': gb_tuned,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
oof_preds = {name: np.zeros_like(y, dtype=float) for name in models}

total_cv_steps = len(models) * 5 + 1
pbar_cv = tqdm(total=total_cv_steps, desc='Project 1: Model CV & GB Tuning', unit='step')

for name, estimator in models.items():
    fold_r2, fold_rmse, fold_mae = [], [], []
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        pbar_cv.set_postfix_str(f'{name} | fold {fold_i}/5')
        model_fold = clone(estimator)
        model_fold.fit(X.iloc[train_idx], y[train_idx])
        preds = model_fold.predict(X.iloc[val_idx])
        oof_preds[name][val_idx] = preds
        fold_r2.append(r2_score(y[val_idx], preds))
        fold_rmse.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        fold_mae.append(mean_absolute_error(y[val_idx], preds))
        pbar_cv.update(1)
    results.append({
        'Model': name,
        'R2_mean': np.mean(fold_r2),
        'R2_std': np.std(fold_r2),
        'RMSE_mean': np.mean(fold_rmse),
        'RMSE_std': np.std(fold_rmse),
        'MAE_mean': np.mean(fold_mae),
    })

pbar_cv.set_postfix_str('Full GB Hyperparameter Search')
best_gb_search = clone(gb_tuned).fit(X, y)
pbar_cv.update(1)
pbar_cv.close()

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
print('\n=== 5-FOLD CROSS-VALIDATION SUMMARY ===')
print(res_df.to_string(index=False))

best_params_df = pd.DataFrame([best_gb_search.best_params_])
best_params_df.to_csv(os.path.join(output_dir, 'gb_best_params.csv'), index=False)
print('\nBest Gradient Boosting Hyperparameters:')
for k, v in best_gb_search.best_params_.items():
    print(f'  {k.replace("model__", "")}: {v}')

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=100)
plt.subplots_adjust(hspace=0.38, wspace=0.25)
for ax, (name, preds) in zip(axes.flat, oof_preds.items()):
    for grp in ['group1', 'group2']:
        mask = (df['group'] == grp).values
        ax.scatter(y[mask], preds[mask], alpha=0.55, s=18, color=palette[grp], label=grp)
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.2)
    r2_val = r2_score(y, preds)
    rmse_val = np.sqrt(mean_squared_error(y, preds))
    ax.set_title(f'{name}\nOOF R² = {r2_val:.3f} | RMSE = {rmse_val:.2f}', fontsize=9.2, fontweight='bold')
    ax.set_xlabel('Actual dependent', fontsize=8.5)
    ax.set_ylabel('Predicted dependent', fontsize=8.5)
    ax.legend(fontsize=7.5)
axes.flat[-1].axis('off')

fig.suptitle('5-Fold Cross-Validation: All Models (Actual vs. Predicted)', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=100)
plt.subplots_adjust(hspace=0.38, wspace=0.25)

for ax, (name, estimator) in zip(axes.flat, tqdm(models.items(), desc='Project 1: Trajectory Plots', unit='plot')):
    fitted_model = clone(estimator).fit(X, y)
    if hasattr(fitted_model, 'best_estimator_'):
        fitted_model = fitted_model.best_estimator_
    for grp in ['group1', 'group2']:
        sub = df[df['group'] == grp]
        ax.scatter(sub['time'], sub['dependent'], alpha=0.35, s=16, color=palette[grp])
        grid_df = pd.DataFrame({
            'group': grp,
            'time': time_grid,
            'var1': sub['var1'].mean(),
            'var2': sub['var2'].mean(),
            'var3': sub['var3'].mean(),
            'var4': sub['var4'].mean(),
        })
        y_line = fitted_model.predict(grid_df)
        ax.plot(time_grid, y_line, color=palette[grp], linewidth=2.3, label=f'{grp} fit')
    ax.set_title(name, fontsize=9.2, fontweight='bold')
    ax.set_xlabel('time', fontsize=8.5)
    ax.set_ylabel('dependent', fontsize=8.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=7.5)
axes.flat[-1].axis('off')

fig.suptitle('Fitted Group Trajectories vs. Time (at Group Predictor Means)', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_group_lines.png'), dpi=150, bbox_inches='tight')
plt.show()