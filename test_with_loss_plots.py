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
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
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

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=100)
plt.subplots_adjust(wspace=0.28)

ax = axes[0]
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
ax.set_title('1. Dependent vs. Time', fontsize=10.5, fontweight='bold')
ax.set_xlabel('time', fontsize=9.5)
ax.set_ylabel('dependent', fontsize=9.5)
ax.set_xticks(range(1, 11))
ax.legend(fontsize=8, loc='upper left')

ax = axes[1]
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
    annot_kws={'size': 9},
)
ax.set_title('2. Pearson Correlation by Group', fontsize=10.5, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

fig.suptitle('Time Dynamics & Group Correlations (n = 400)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_time_corr.png'), dpi=150, bbox_inches='tight')
plt.show()

vars_to_plot = ['var1', 'var2', 'var3', 'var4']
titles = [
    '1. Dependent vs. var1',
    '2. Dependent vs. var2',
    '3. Dependent vs. var3',
    '4. Dependent vs. var4',
]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=100)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

for ax, var, title in zip(axes.flat, vars_to_plot, titles):
    for grp, sub in df.groupby('group'):
        ax.scatter(sub[var], sub['dependent'], alpha=0.45, s=18, color=palette[grp], label=grp)
        m, b = np.polyfit(sub[var], sub['dependent'], 1)
        x_vals = np.linspace(sub[var].min(), sub[var].max(), 100)
        ax.plot(x_vals, m * x_vals + b, color=palette[grp], linewidth=1.8, linestyle='--')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(var, fontsize=9)
    ax.set_ylabel('dependent', fontsize=9)
    ax.legend(fontsize=8)

fig.suptitle('Exploratory Data Analysis: Predictors vs. Dependent (n = 400)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_predictors.png'), dpi=150, bbox_inches='tight')
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

# outer validation: 5-fold CV stratified by group x time (each fold keeps both groups and all time points)
strat_labels = (df['group'].astype(str) + '_' + df['time'].astype(str)).values
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
oof_preds = {name: np.zeros_like(y, dtype=float) for name in models}

total_cv_steps = len(models) * 5 + 1
pbar_cv = tqdm(total=total_cv_steps, desc='Project 1: Model CV & GB Tuning', unit='step')

train_rmse_folds = {}
val_rmse_folds = {}

for name, estimator in models.items():
    fold_r2, fold_rmse, fold_mae, fold_train_rmse = [], [], [], []
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, strat_labels), start=1):
        pbar_cv.set_postfix_str(f'{name} | fold {fold_i}/5')
        model_fold = clone(estimator)
        model_fold.fit(X.iloc[train_idx], y[train_idx])
        preds = model_fold.predict(X.iloc[val_idx])
        train_preds = model_fold.predict(X.iloc[train_idx])
        oof_preds[name][val_idx] = preds
        fold_r2.append(r2_score(y[val_idx], preds))
        fold_rmse.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        fold_mae.append(mean_absolute_error(y[val_idx], preds))
        fold_train_rmse.append(np.sqrt(mean_squared_error(y[train_idx], train_preds)))
        pbar_cv.update(1)
    train_rmse_folds[name] = fold_train_rmse
    val_rmse_folds[name] = fold_rmse
    results.append({
        'Model': name,
        'R2_mean': np.mean(fold_r2),
        'R2_std': np.std(fold_r2),
        'RMSE_mean': np.mean(fold_rmse),
        'RMSE_std': np.std(fold_rmse),
        'MAE_mean': np.mean(fold_mae),
        'Train_RMSE_mean': np.mean(fold_train_rmse),
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

# ---------- A) Train vs. test (validation) RMSE for every model ----------
names = list(models.keys())
tr_mean = [np.mean(train_rmse_folds[n]) for n in names]
tr_std = [np.std(train_rmse_folds[n]) for n in names]
va_mean = [np.mean(val_rmse_folds[n]) for n in names]
va_std = [np.std(val_rmse_folds[n]) for n in names]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=100, gridspec_kw={'width_ratios': [1.3, 1]})

ax = axes[0]
x_pos = np.arange(len(names))
w = 0.36
ax.bar(x_pos - w / 2, tr_mean, w, yerr=tr_std, capsize=3, color='#4c72b0', alpha=0.85, label='Train RMSE')
ax.bar(x_pos + w / 2, va_mean, w, yerr=va_std, capsize=3, color='#dd8452', alpha=0.85, label='Test (val) RMSE')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=8)
ax.set_ylabel('RMSE')
ax.set_title('Train vs. Test RMSE (mean ± std over 5 folds)', fontsize=10.5, fontweight='bold')
ax.legend()

ax = axes[1]
for n, c in zip(names, sns.color_palette('tab10', len(names))):
    ax.plot(range(1, 6), train_rmse_folds[n], '--o', color=c, alpha=0.6, markersize=4)
    ax.plot(range(1, 6), val_rmse_folds[n], '-o', color=c, label=n, markersize=4)
ax.set_xticks(range(1, 6))
ax.set_xlabel('CV fold')
ax.set_ylabel('RMSE')
ax.set_title('Per-fold RMSE (dashed = train, solid = test)', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_test_rmse.png'), dpi=150, bbox_inches='tight')
plt.show()


# ---------- B) Loss curves vs. boosting iteration (Gradient Boosting) ----------
def gb_loss_curves(pipe_template, X, y, kf):
    """Train/test RMSE after every boosting stage, for each CV fold."""
    tr_curves, te_curves = [], []
    for tr_idx, te_idx in kf.split(X, strat_labels):
        pipe = clone(pipe_template).fit(X.iloc[tr_idx], y[tr_idx])
        prep_step, gb = pipe.named_steps['prep'], pipe.named_steps['model']
        Xtr = prep_step.transform(X.iloc[tr_idx])
        Xte = prep_step.transform(X.iloc[te_idx])
        tr_curves.append([np.sqrt(mean_squared_error(y[tr_idx], p)) for p in gb.staged_predict(Xtr)])
        te_curves.append([np.sqrt(mean_squared_error(y[te_idx], p)) for p in gb.staged_predict(Xte)])
    return np.array(tr_curves), np.array(te_curves)


# tuned pipeline = best hyperparameters found by the search (unfitted copy)
gb_tuned_pipe = clone(best_gb_search.best_estimator_)

gb_variants = {
    '4. Gradient Boosting (Default)': gb_default_pipe,
    '5. Gradient Boosting (Tuned)': gb_tuned_pipe,
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=100, sharey=True)
for ax, (name, pipe) in zip(axes, gb_variants.items()):
    tr, te = gb_loss_curves(pipe, X, y, kf)
    it = np.arange(1, tr.shape[1] + 1)
    for curve, color, lab in [(tr, '#4c72b0', 'Train'), (te, '#dd8452', 'Test (val)')]:
        m, s = curve.mean(0), curve.std(0)
        ax.plot(it, m, color=color, linewidth=2, label=lab)
        ax.fill_between(it, m - s, m + s, color=color, alpha=0.2)
    best_it = te.mean(0).argmin() + 1
    ax.axvline(best_it, color='gray', linestyle=':', label=f'min test RMSE @ {best_it}')
    ax.set_title(name, fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Boosting iteration')
    ax.legend(fontsize=8)
axes[0].set_ylabel('RMSE (loss)')

fig.suptitle('Gradient Boosting: Train vs. Test Loss Curves (mean ± std, 5 folds)', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gb_loss_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

fitted_models = {}
for name, estimator in tqdm(models.items(), desc='Project 1: Fitting Full Models', unit='model'):
    if name == '5. Gradient Boosting (Tuned)':
        fitted_models[name] = best_gb_search.best_estimator_
    else:
        fitted_models[name] = clone(estimator).fit(X, y)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=100)
plt.subplots_adjust(hspace=0.38, wspace=0.25)

for ax, (name, fitted_model) in zip(axes.flat, fitted_models.items()):
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

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=100)
plt.subplots_adjust(hspace=0.38, wspace=0.25)

for ax, (name, fitted_model) in zip(axes.flat, fitted_models.items()):
    for grp in ['group1', 'group2']:
        sub = df[df['group'] == grp]
        offset = -0.15 if grp == 'group1' else 0.15
        positions = np.arange(1, 11) + offset
        box_data = [sub[sub['time'] == t]['dependent'].values for t in range(1, 11)]

        ax.boxplot(
            box_data,
            positions=positions,
            widths=0.24,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(
                marker='D',
                markeredgecolor='black',
                markerfacecolor='white',
                markersize=4,
                zorder=5,
            ),
            medianprops=dict(color='black', linewidth=1.2),
            boxprops=dict(
                facecolor=palette[grp],
                alpha=0.45,
                edgecolor=palette[grp],
                linewidth=1,
            ),
            whiskerprops=dict(color=palette[grp], linewidth=1),
            capprops=dict(color=palette[grp], linewidth=1),
            flierprops=dict(
                marker='o',
                markersize=3,
                markerfacecolor=palette[grp],
                markeredgecolor='none',
                alpha=0.6,
            ),
        )

        grid_df = pd.DataFrame({
            'group': grp,
            'time': time_grid,
            'var1': sub['var1'].mean(),
            'var2': sub['var2'].mean(),
            'var3': sub['var3'].mean(),
            'var4': sub['var4'].mean(),
        })
        y_line = fitted_model.predict(grid_df)
        ax.plot(time_grid, y_line, color=palette[grp], linewidth=2.2, label=f'{grp} fit', zorder=4)

    ax.scatter([], [], marker='D', edgecolors='black', facecolors='white', s=22, label='group mean')
    ax.set_title(name, fontsize=9.2, fontweight='bold')
    ax.set_xlabel('time', fontsize=8.5)
    ax.set_ylabel('dependent', fontsize=8.5)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(range(1, 11))
    ax.set_xlim(0.4, 10.6)
    ax.legend(fontsize=7.2, loc='upper left')
axes.flat[-1].axis('off')

fig.suptitle('Fitted Group Trajectories vs. Time with Per-Second Distributions & Means', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'candlestick_models_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()