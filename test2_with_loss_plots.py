import os
import warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    normalized_mutual_info_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

output_dir = os.path.join('images', 'project2')
os.makedirs(output_dir, exist_ok=True)

sns.set_theme(style='whitegrid', context='notebook')

sat = pd.read_csv('satellite_image.csv')
lbl = pd.read_csv('labels.csv')


def build_features(df_in):
    X = df_in[[f'band{i}' for i in range(1, 9)]].copy()
    eps = 1e-8
    X['NDVI'] = (X['band8'] - X['band4']) / (X['band8'] + X['band4'] + eps)
    X['NDWI'] = (X['band3'] - X['band8']) / (X['band3'] + X['band8'] + eps)
    X['RedEdge_Ratio'] = X['band7'] / (X['band5'] + eps)
    X['Brightness'] = X[[f'band{i}' for i in range(1, 9)]].mean(axis=1)
    return X


X_lbl = build_features(lbl)
y_lbl = lbl['label'].values
classes = sorted(lbl['label'].unique())

class_styles = {
    'grass': {'color': '#7CFC00', 'marker': 'o'},
    'tree': {'color': '#006400', 'marker': '^'},
    'water': {'color': '#00BFFF', 'marker': 's'},
    'soil': {'color': '#D2691E', 'marker': 'D'},
    'road1': {'color': '#FFD700', 'marker': 'v'},
    'road2': {'color': '#FF8C00', 'marker': '<'},
    'roof1': {'color': '#FF1493', 'marker': 'P'},
    'roof2': {'color': '#DC143C', 'marker': 'X'},
    'roof3': {'color': '#9400D3', 'marker': '*'},
    'shadow': {'color': '#2F4F4F', 'marker': 'h'},
}

fig_eda, axes_eda = plt.subplots(1, 3, figsize=(13.8, 4.2), dpi=100)

counts = lbl['label'].value_counts().reindex(classes)
sns.barplot(
    x=counts.values,
    y=counts.index,
    hue=counts.index,
    palette=[class_styles[c]['color'] for c in classes],
    edgecolor='black',
    linewidth=0.6,
    legend=False,
    ax=axes_eda[0],
)
for i, v in enumerate(counts.values):
    axes_eda[0].text(v + 0.8, i, str(v), va='center', fontsize=8)
axes_eda[0].set_xlim(0, counts.max() * 1.16)
axes_eda[0].set_title('1. Class Sample Counts (n = 319)', fontsize=10, fontweight='bold')
axes_eda[0].set_xlabel('Number of Labeled Pixels', fontsize=9)
axes_eda[0].set_ylabel('Land Cover Class', fontsize=9)

band_cols = [f'band{i}' for i in range(1, 9)]
mean_spectra = lbl.groupby('label')[band_cols].mean().reindex(classes)
for cls in classes:
    axes_eda[1].plot(
        range(1, 9),
        mean_spectra.loc[cls].values,
        marker='o',
        markersize=4,
        linewidth=1.8,
        color=class_styles[cls]['color'],
        label=cls,
    )
axes_eda[1].set_title('2. Mean Spectral Signatures (Bands 1–8)', fontsize=10, fontweight='bold')
axes_eda[1].set_xlabel('Spectral Band', fontsize=9)
axes_eda[1].set_ylabel('Mean Reflectance / Brightness', fontsize=9)
axes_eda[1].set_xticks(range(1, 9))
axes_eda[1].set_xticklabels([f'B{i}' for i in range(1, 9)], fontsize=8.5)
axes_eda[1].legend(fontsize=7, ncol=2, loc='upper left')

eda_scatter_df = X_lbl[['NDVI', 'NDWI']].copy()
eda_scatter_df['label'] = y_lbl
for cls in classes:
    sub = eda_scatter_df[eda_scatter_df['label'] == cls]
    axes_eda[2].scatter(
        sub['NDVI'],
        sub['NDWI'],
        c=class_styles[cls]['color'],
        marker=class_styles[cls]['marker'],
        s=32,
        edgecolors='black',
        linewidths=0.4,
        alpha=0.85,
        label=cls,
    )
axes_eda[2].set_title('3. Spectral Indices Space (NDVI vs. NDWI)', fontsize=10, fontweight='bold')
axes_eda[2].set_xlabel('NDVI = (B8 - B4) / (B8 + B4)', fontsize=9)
axes_eda[2].set_ylabel('NDWI = (B3 - B8) / (B3 + B8)', fontsize=9)

fig_eda.suptitle('Exploratory Data Analysis: Reference Labels & Spectral Profiles', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_spectral_data.png'), dpi=150, bbox_inches='tight')
plt.show()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_models = {
    '1. Logistic Regression': Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    '2. KNN (k=5)': Pipeline([
        ('scale', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5)),
    ]),
    '3. SVM (RBF Kernel)': Pipeline([
        ('scale', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42)),
    ]),
    '4. Random Forest': RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
}

metrics_rows = []
oof_base = {}
loss_records = []


def safe_logloss(model, X_, y_true):
    if not hasattr(model, 'predict_proba'):
        return np.nan
    return log_loss(y_true, model.predict_proba(X_), labels=model.classes_)


def record_fold_loss(stage, name, fold_i, model, train_idx, val_idx):
    Xtr, ytr = X_lbl.iloc[train_idx], y_lbl[train_idx]
    Xva, yva = X_lbl.iloc[val_idx], y_lbl[val_idx]
    loss_records.append({
        'Stage': stage,
        'Model': name,
        'Fold': fold_i,
        'Train_Error': 1.0 - accuracy_score(ytr, model.predict(Xtr)),
        'Val_Error': 1.0 - accuracy_score(yva, model.predict(Xva)),
        'Train_LogLoss': safe_logloss(model, Xtr, ytr),
        'Val_LogLoss': safe_logloss(model, Xva, yva),
    })


pbar_base = tqdm(total=len(base_models) * 5, desc='Part A: Baseline CV', unit='fold')
for name, clf in base_models.items():
    y_pred = np.empty_like(y_lbl)
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_lbl, y_lbl), start=1):
        pbar_base.set_postfix_str(name)
        m_fold = clone(clf)
        m_fold.fit(X_lbl.iloc[train_idx], y_lbl[train_idx])
        y_pred[val_idx] = m_fold.predict(X_lbl.iloc[val_idx])
        record_fold_loss('Baseline', name, fold_i, m_fold, train_idx, val_idx)
        pbar_base.update(1)
    oof_base[name] = y_pred
    acc = accuracy_score(y_lbl, y_pred)
    kappa = cohen_kappa_score(y_lbl, y_pred)
    macro_f1 = f1_score(y_lbl, y_pred, average='macro')
    weighted_f1 = f1_score(y_lbl, y_pred, average='weighted')
    metrics_rows.append({
        'Stage': 'Baseline',
        'Model': name,
        'Accuracy': acc,
        'Error_Rate': 1.0 - acc,
        'Cohen_Kappa': kappa,
        'Macro_F1': macro_f1,
        'Weighted_F1': weighted_f1,
    })
pbar_base.close()

fig_cm, axes_cm = plt.subplots(2, 2, figsize=(11.8, 7.6), dpi=100)
plt.subplots_adjust(hspace=0.38, wspace=0.22)
for idx, (name, y_pred) in enumerate(oof_base.items()):
    ax = axes_cm.flat[idx]
    cm = confusion_matrix(y_lbl, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot_mat = np.where(cm > 0, cm.astype(str), '')
    row = metrics_rows[idx]
    sns.heatmap(
        cm_norm,
        annot=annot_mat,
        fmt='',
        cmap='Blues',
        vmin=0,
        vmax=100,
        xticklabels=classes,
        yticklabels=classes if idx % 2 == 0 else False,
        cbar=idx % 2 == 1,
        cbar_kws={'label': 'Recall (%)'} if idx % 2 == 1 else None,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 7.5},
    )
    ax.set_title(
        f"{name}\nAcc = {row['Accuracy']*100:.1f}% | κ = {row['Cohen_Kappa']:.3f} | F1 = {row['Macro_F1']:.3f}",
        fontsize=9.2,
        fontweight='bold',
    )
    ax.set_xlabel('Predicted', fontsize=8.5)
    if idx % 2 == 0:
        ax.set_ylabel('True Class', fontsize=8.5)
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

fig_cm.suptitle('Baseline Classifiers: 5-Fold Stratified CV Confusion Matrices', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'models_baseline_cm.png'), dpi=150, bbox_inches='tight')
plt.show()

nx = sat['x'].nunique()
ny = sat['y'].nunique()
sat_sorted = sat.sort_values(by=['y', 'x'], ascending=[False, True]).reset_index(drop=True)


def stretch_band(arr, low_pct=2, high_pct=98):
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    return np.clip((arr - lo) / (hi - lo), 0, 1)


R = stretch_band(sat_sorted['band4'].values).reshape(ny, nx)
G = stretch_band(sat_sorted['band3'].values).reshape(ny, nx)
B = stretch_band(sat_sorted['band2'].values).reshape(ny, nx)
rgb_img = np.dstack([R, G, B])
extent = [sat['x'].min(), sat['x'].max(), sat['y'].min(), sat['y'].max()]

fig_tc, ax_tc = plt.subplots(figsize=(7.6, 5.6), dpi=100)
ax_tc.imshow(rgb_img, extent=extent, origin='upper', aspect='equal')
for cls in classes:
    sub = lbl[lbl['label'] == cls]
    ax_tc.scatter(
        sub['x'],
        sub['y'],
        c=class_styles[cls]['color'],
        marker=class_styles[cls]['marker'],
        s=38,
        edgecolors='black',
        linewidths=0.65,
        label=f'{cls} (n={len(sub)})',
        zorder=4,
    )
ax_tc.set_title('True-Color Raster (R: Band 4, G: Band 3, B: Band 2) with Reference Labels', fontsize=10, fontweight='bold')
ax_tc.set_xlabel('Easting X (m)', fontsize=9)
ax_tc.set_ylabel('Northing Y (m)', fontsize=9)
ax_tc.ticklabel_format(style='plain', useOffset=False)
ax_tc.tick_params(labelsize=8)
ax_tc.legend(
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    frameon=True,
    facecolor='white',
    title='Reference Labels',
    title_fontsize=8.5,
    fontsize=7.8,
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'raster_true_color_labels.png'), dpi=150, bbox_inches='tight')
plt.show()

X_sat = build_features(sat_sorted)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
cmap = ListedColormap([class_styles[cls]['color'] for cls in classes])
norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5, 1), len(classes))

legend_patches = [
    mpatches.Patch(
        facecolor=class_styles[cls]['color'],
        edgecolor='black',
        linewidth=0.5,
        label=cls,
    )
    for cls in classes
]

fig_maps, axes_maps = plt.subplots(2, 2, figsize=(12.2, 7.6), dpi=100)
plt.subplots_adjust(hspace=0.32, wspace=0.18)
for idx, (name, clf) in enumerate(tqdm(base_models.items(), desc='Part A: Baseline Maps', unit='map')):
    ax = axes_maps.flat[idx]
    clf_fit = clone(clf).fit(X_lbl, y_lbl)
    preds_sat = clf_fit.predict(X_sat)
    pred_grid = pd.Series(preds_sat).map(class_to_idx).values.reshape(ny, nx)
    acc_val = metrics_rows[idx]['Accuracy'] * 100
    ax.imshow(
        pred_grid,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin='upper',
        aspect='equal',
        interpolation='nearest',
    )
    ax.set_title(f'{name} (CV Acc = {acc_val:.1f}%)', fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Easting X (m)', fontsize=8.5)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    if idx % 2 == 0:
        ax.set_ylabel('Northing Y (m)', fontsize=8.5)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    else:
        ax.set_yticks([])
    ax.tick_params(labelsize=7.5)

axes_maps[0, 1].legend(
    handles=legend_patches,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    facecolor='white',
    title='Land Cover',
    title_fontsize=8.5,
    fontsize=7.8,
)

fig_maps.suptitle('Baseline Predicted Land Cover Maps (180,947 Pixels)', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_maps_baseline.png'), dpi=150, bbox_inches='tight')
plt.show()

inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

tuned_models = {
    '1. Logistic Regression': RandomizedSearchCV(
        estimator=Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        param_distributions={
            'clf__C': np.logspace(-2, 1.5, 25),
            'clf__class_weight': [None, 'balanced'],
        },
        n_iter=25,
        scoring='f1_macro',
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
    ),
    '2. KNN (Tuned)': RandomizedSearchCV(
        estimator=Pipeline([
            ('scale', StandardScaler()),
            ('clf', KNeighborsClassifier()),
        ]),
        param_distributions={
            'clf__n_neighbors': [5, 7, 9, 11, 15, 19],
            'clf__weights': ['uniform'],
            'clf__metric': ['euclidean', 'manhattan', 'minkowski'],
            'clf__p': [1, 2],
        },
        n_iter=25,
        scoring='f1_macro',
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
    ),
    '3. SVM (RBF Kernel)': RandomizedSearchCV(
        estimator=Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVC(kernel='rbf', random_state=42)),
        ]),
        param_distributions={
            'clf__C': np.logspace(-1, 1.5, 25),
            'clf__gamma': ['scale', 'auto', 0.005, 0.01, 0.03, 0.05, 0.1],
            'clf__class_weight': [None, 'balanced'],
        },
        n_iter=35,
        scoring='f1_macro',
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
    ),
    '4. Random Forest': RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions={
            'n_estimators': [200, 400, 600],
            'max_depth': [4, 6, 8, 10],
            'min_samples_split': [5, 8, 12, 16],
            'min_samples_leaf': [3, 5, 7, 10],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'max_samples': [0.7, 0.85, None],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
        },
        n_iter=40,
        scoring='f1_macro',
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
    ),
}

oof_tuned = {}
best_params_records = []
best_fitted_estimators = {}

total_tune_steps = len(tuned_models) * (5 + 1)
pbar_tune = tqdm(total=total_tune_steps, desc='Part B: Hyperparameter Tuning', unit='step')

for name, search_clf in tuned_models.items():
    y_pred = np.empty_like(y_lbl)
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_lbl, y_lbl), start=1):
        pbar_tune.set_postfix_str(f'{name} | fold {fold_i}/5')
        m_fold = clone(search_clf)
        m_fold.fit(X_lbl.iloc[train_idx], y_lbl[train_idx])
        y_pred[val_idx] = m_fold.predict(X_lbl.iloc[val_idx])
        record_fold_loss('Tuned', name, fold_i, m_fold, train_idx, val_idx)
        pbar_tune.update(1)
    oof_tuned[name] = y_pred
    acc = accuracy_score(y_lbl, y_pred)
    kappa = cohen_kappa_score(y_lbl, y_pred)
    macro_f1 = f1_score(y_lbl, y_pred, average='macro')
    weighted_f1 = f1_score(y_lbl, y_pred, average='weighted')
    metrics_rows.append({
        'Stage': 'Tuned',
        'Model': name,
        'Accuracy': acc,
        'Error_Rate': 1.0 - acc,
        'Cohen_Kappa': kappa,
        'Macro_F1': macro_f1,
        'Weighted_F1': weighted_f1,
    })
    pbar_tune.set_postfix_str(f'{name} | full fit')
    full_search = clone(search_clf).fit(X_lbl, y_lbl)
    best_fitted_estimators[name] = full_search.best_estimator_
    best_params_records.append({'Model': name, 'Best_Params': str(full_search.best_params_)})
    pbar_tune.update(1)

pbar_tune.close()

summary_df = pd.DataFrame(metrics_rows)
summary_df.to_csv(os.path.join(output_dir, 'metrics_baseline_vs_tuned.csv'), index=False)
pd.DataFrame(best_params_records).to_csv(os.path.join(output_dir, 'best_hyperparameters.csv'), index=False)

print('\n=== SUMMARY TABLE: BASELINE VS. TUNED ===')
print(summary_df.to_string(index=False))

loss_df = pd.DataFrame(loss_records)
loss_df.to_csv(os.path.join(output_dir, 'train_val_loss_per_fold.csv'), index=False)

fig_loss, axes_loss = plt.subplots(2, 2, figsize=(13.8, 8.4), dpi=100)
plt.subplots_adjust(hspace=0.42, wspace=0.18)
metric_specs = [
    ('Error', 'Error rate (0-1 loss = 1 - accuracy)'),
    ('LogLoss', 'Log-loss (cross-entropy, log scale)'),
]
for col, stage in enumerate(['Baseline', 'Tuned']):
    sub_stage = loss_df[loss_df['Stage'] == stage]
    for row, (metric, ylabel) in enumerate(metric_specs):
        ax = axes_loss[row, col]
        agg = sub_stage.groupby('Model', sort=False)[[f'Train_{metric}', f'Val_{metric}']].agg(['mean', 'std'])
        names_stage = list(agg.index)
        x_pos = np.arange(len(names_stage))
        w = 0.36
        tr_m, tr_s = agg[(f'Train_{metric}', 'mean')].values, agg[(f'Train_{metric}', 'std')].values
        va_m, va_s = agg[(f'Val_{metric}', 'mean')].values, agg[(f'Val_{metric}', 'std')].values
        ax.bar(
            x_pos - w / 2,
            np.nan_to_num(tr_m),
            w,
            yerr=np.nan_to_num(tr_s),
            capsize=3,
            color='#4c72b0',
            alpha=0.85,
            label='Train',
        )
        ax.bar(
            x_pos + w / 2,
            np.nan_to_num(va_m),
            w,
            yerr=np.nan_to_num(va_s),
            capsize=3,
            color='#dd8452',
            alpha=0.85,
            label='Validation',
        )
        for i in range(len(names_stage)):
            if np.isnan(tr_m[i]):
                ax.text(
                    x_pos[i],
                    0.5 if metric == 'LogLoss' else 0.02,
                    'n/a\n(no probabilities)',
                    ha='center',
                    va='bottom',
                    fontsize=7.5,
                    color='gray',
                )
        if metric == 'LogLoss':
            ax.set_yscale('log')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([n.replace(' (', '\n(') for n in names_stage], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(
            f'{stage}: {metric.replace("LogLoss", "Log-loss")} (mean ± std, 5 folds)',
            fontsize=10,
            fontweight='bold',
        )
        ax.legend(fontsize=8)

fig_loss.suptitle('Train vs. Validation Loss: Baseline and Tuned Classifiers', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_val_loss.png'), dpi=150, bbox_inches='tight')
plt.show()


def error_learning_curve(est):
    sizes, tr_sc, va_sc = learning_curve(
        est,
        X_lbl,
        y_lbl,
        train_sizes=np.linspace(0.2, 1.0, 8),
        cv=skf,
        scoring='accuracy',
        shuffle=True,
        random_state=42,
    )
    return sizes, 1.0 - tr_sc, 1.0 - va_sc


tuned_estimators = list(best_fitted_estimators.values())
fig_lc, axes_lc = plt.subplots(2, 2, figsize=(12.6, 8.0), dpi=100, sharey=True)
plt.subplots_adjust(hspace=0.32, wspace=0.12)
for idx, (name, clf) in enumerate(tqdm(base_models.items(), desc='Learning curves', unit='model')):
    ax = axes_lc.flat[idx]
    for est, ls, tag in [(clf, '-', 'baseline'), (tuned_estimators[idx], '--', 'tuned regularized')]:
        sizes, tr_err, va_err = error_learning_curve(clone(est))
        for curve, color, lab in [(tr_err, '#4c72b0', 'train'), (va_err, '#dd8452', 'validation')]:
            m, sd = curve.mean(1), curve.std(1)
            ax.plot(sizes, m, ls, color=color, linewidth=2, marker='o', markersize=3.5, label=f'{lab} ({tag})')
            ax.fill_between(sizes, m - sd, m + sd, color=color, alpha=0.12)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Training set size (labeled pixels)', fontsize=8.5)
    if idx % 2 == 0:
        ax.set_ylabel('Error rate (1 - accuracy)', fontsize=8.5)
    ax.legend(fontsize=7.2)

fig_lc.suptitle('Learning Curves: Train vs. Validation Error (mean ± std, 5 folds)', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

fig_cm_t, axes_cm_t = plt.subplots(2, 2, figsize=(11.8, 7.6), dpi=100)
plt.subplots_adjust(hspace=0.38, wspace=0.22)
tuned_summary = summary_df[summary_df['Stage'] == 'Tuned'].reset_index(drop=True)
for idx, (name, y_pred) in enumerate(oof_tuned.items()):
    ax = axes_cm_t.flat[idx]
    cm = confusion_matrix(y_lbl, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot_mat = np.where(cm > 0, cm.astype(str), '')
    row = tuned_summary.iloc[idx]
    sns.heatmap(
        cm_norm,
        annot=annot_mat,
        fmt='',
        cmap='Greens',
        vmin=0,
        vmax=100,
        xticklabels=classes,
        yticklabels=classes if idx % 2 == 0 else False,
        cbar=idx % 2 == 1,
        cbar_kws={'label': 'Recall (%)'} if idx % 2 == 1 else None,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 7.5},
    )
    ax.set_title(
        f"{name}\nAcc = {row['Accuracy']*100:.1f}% | κ = {row['Cohen_Kappa']:.3f} | F1 = {row['Macro_F1']:.3f}",
        fontsize=9.2,
        fontweight='bold',
    )
    ax.set_xlabel('Predicted', fontsize=8.5)
    if idx % 2 == 0:
        ax.set_ylabel('True Class', fontsize=8.5)
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

fig_cm_t.suptitle('Tuned Classifiers: 5-Fold Nested Stratified CV Confusion Matrices', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'models_tuned_cm.png'), dpi=150, bbox_inches='tight')
plt.show()

fig_maps_t, axes_maps_t = plt.subplots(2, 2, figsize=(12.2, 7.6), dpi=100)
plt.subplots_adjust(hspace=0.32, wspace=0.18)
for idx, (name, best_est) in enumerate(tqdm(best_fitted_estimators.items(), desc='Part B: Tuned Maps', unit='map')):
    ax = axes_maps_t.flat[idx]
    preds_sat = best_est.predict(X_sat)
    pred_grid = pd.Series(preds_sat).map(class_to_idx).values.reshape(ny, nx)
    acc_val = tuned_summary.iloc[idx]['Accuracy'] * 100
    ax.imshow(
        pred_grid,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin='upper',
        aspect='equal',
        interpolation='nearest',
    )
    ax.set_title(f'{name} (Tuned CV Acc = {acc_val:.1f}%)', fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Easting X (m)', fontsize=8.5)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    if idx % 2 == 0:
        ax.set_ylabel('Northing Y (m)', fontsize=8.5)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    else:
        ax.set_yticks([])
    ax.tick_params(labelsize=7.5)

axes_maps_t[0, 1].legend(
    handles=legend_patches,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    facecolor='white',
    title='Land Cover',
    title_fontsize=8.5,
    fontsize=7.8,
)

fig_maps_t.suptitle('Tuned Predicted Land Cover Maps (180,947 Pixels)', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_maps_tuned.png'), dpi=150, bbox_inches='tight')
plt.show()

scaler_km = StandardScaler()
X_sat_scaled = scaler_km.fit_transform(X_sat)
X_lbl_scaled = scaler_km.transform(X_lbl)

kmeans = KMeans(n_clusters=len(classes), n_init=20, random_state=42)
sat_clusters = kmeans.fit_predict(X_sat_scaled)
lbl_clusters = kmeans.predict(X_lbl_scaled)

cluster_to_class = {}
for c_id in range(len(classes)):
    mask = lbl_clusters == c_id
    if np.any(mask):
        cluster_to_class[c_id] = pd.Series(y_lbl[mask]).mode()[0]
    else:
        cluster_to_class[c_id] = 'shadow'

lbl_km_mapped = np.array([cluster_to_class[c] for c in lbl_clusters])
sat_km_mapped = np.array([cluster_to_class[c] for c in sat_clusters])

km_acc = accuracy_score(y_lbl, lbl_km_mapped)
km_ari = adjusted_rand_score(y_lbl, lbl_clusters)
km_nmi = normalized_mutual_info_score(y_lbl, lbl_clusters)

print(f'\nK-Means (K=10) Unsupervised Metrics on Reference Set:')
print(f'  Purity / Mapped Accuracy : {km_acc*100:.2f}%')
print(f'  Adjusted Rand Index (ARI): {km_ari:.4f}')
print(f'  Normalized Mutual Info   : {km_nmi:.4f}')

fig_km, (ax_km1, ax_km2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=100)
raw_cluster_grid = sat_clusters.reshape(ny, nx)
ax_km1.imshow(
    raw_cluster_grid,
    cmap='tab10',
    extent=extent,
    origin='upper',
    aspect='equal',
    interpolation='nearest',
)
ax_km1.set_title(f'Unsupervised K-Means (10 Raw Clusters)\nARI = {km_ari:.3f} | NMI = {km_nmi:.3f}', fontsize=9.8, fontweight='bold')
ax_km1.set_xlabel('Easting X (m)', fontsize=8.5)
ax_km1.set_ylabel('Northing Y (m)', fontsize=8.5)
ax_km1.ticklabel_format(style='plain', useOffset=False)
ax_km1.tick_params(labelsize=7.5)

mapped_grid = pd.Series(sat_km_mapped).map(class_to_idx).values.reshape(ny, nx)
ax_km2.imshow(
    mapped_grid,
    cmap=cmap,
    norm=norm,
    extent=extent,
    origin='upper',
    aspect='equal',
    interpolation='nearest',
)
ax_km2.set_title(f'K-Means Mapped to Land Cover Classes\n(Majority-Vote Purity = {km_acc*100:.1f}%)', fontsize=9.8, fontweight='bold')
ax_km2.set_xlabel('Easting X (m)', fontsize=8.5)
ax_km2.ticklabel_format(axis='x', style='plain', useOffset=False)
ax_km2.set_yticks([])
ax_km2.tick_params(labelsize=7.5)
ax_km2.legend(
    handles=legend_patches,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    facecolor='white',
    title='Land Cover',
    title_fontsize=8.5,
    fontsize=7.8,
)

fig_km.suptitle('Unsupervised Clustering Approach: K-Means on 180,947 Satellite Pixels', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kmeans_unsupervised_map.png'), dpi=150, bbox_inches='tight')
plt.show()