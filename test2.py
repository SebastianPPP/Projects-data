import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# 1. Wczytanie danych
sat = pd.read_csv('satellite_image.csv')
lbl = pd.read_csv('labels.csv')


# 2. Inżynieria cech: 8 pasm spektralnych + indeksy teledetekcyjne (bez współrzędnych x, y!)
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

# 3. Walidacja krzyżowa Stratified 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

y_pred_cv = cross_val_predict(rf, X_lbl, y_lbl, cv=skf)

overall_acc = accuracy_score(y_lbl, y_pred_cv)
kappa = cohen_kappa_score(y_lbl, y_pred_cv)
macro_f1 = f1_score(y_lbl, y_pred_cv, average='macro')
weighted_f1 = f1_score(y_lbl, y_pred_cv, average='weighted')

print(f'Overall Accuracy : {overall_acc:.4f} ({overall_acc*100:.2f}%)')
print(f"Cohen's Kappa    : {kappa:.4f}")
print(f'Macro F1-Score   : {macro_f1:.4f}')
print(f'Weighted F1-Score: {weighted_f1:.4f}\n')
print('Per-Class Classification Report:')
print(classification_report(y_lbl, y_pred_cv, digits=4))

# 4. Trening finalnego modelu na wszystkich 319 próbkach i predykcja dla całego rastra
rf.fit(X_lbl, y_lbl)
X_sat = build_features(sat)
sat['pred_label'] = rf.predict(X_sat)

# 5. Rekonstrukcja siatki 2D rastra (403 wiersze x 449 kolumn)
nx = sat['x'].nunique()  # 449
ny = sat['y'].nunique()  # 403

# Sortowanie od góry do dołu (y malejąco) oraz od lewej do prawej (x rosnąco)
sat_sorted = sat.sort_values(
    by=['y', 'x'], ascending=[False, True]
).reset_index(drop=True)


def stretch_band(arr, low_pct=2, high_pct=98):
  lo, hi = np.percentile(arr, [low_pct, high_pct])
  return np.clip((arr - lo) / (hi - lo), 0, 1)


R = stretch_band(sat_sorted['band4'].values).reshape(ny, nx)
G = stretch_band(sat_sorted['band3'].values).reshape(ny, nx)
B = stretch_band(sat_sorted['band2'].values).reshape(ny, nx)
rgb_img = np.dstack([R, G, B])

extent = [sat['x'].min(), sat['x'].max(), sat['y'].min(), sat['y'].max()]

# Spójna mapa kolorów i znaczników dla 10 klas pokrycia terenu
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

# ==============================================================================
# WYKRES 1: Macierz pomyłek (Confusion Matrix)
# ==============================================================================
cm = confusion_matrix(y_lbl, y_pred_cv, labels=classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

annot_matrix = np.empty_like(cm, dtype=object)
for i in range(len(classes)):
  for j in range(len(classes)):
    if cm[i, j] > 0:
      annot_matrix[i, j] = f'{cm[i, j]}\n({cm_norm[i, j]:.0f}%)'
    else:
      annot_matrix[i, j] = '0'

fig_cm, ax_cm = plt.subplots(figsize=(9.5, 7.5), dpi=180)
sns.heatmap(
    cm_norm,
    annot=annot_matrix,
    fmt='',
    cmap='Blues',
    vmin=0,
    vmax=100,
    xticklabels=classes,
    yticklabels=classes,
    cbar_kws={'label': 'Row-Normalized Accuracy (%)'},
    linewidths=0.6,
    linecolor='white',
    ax=ax_cm,
)
ax_cm.set_title(
    'Out-of-Fold Confusion Matrix (5-Fold Stratified CV, n = 319)\n'
    f"Overall Accuracy = {overall_acc*100:.2f}% | Cohen's κ = {kappa:.3f} |"
    f' Macro F1 = {macro_f1:.3f}',
    fontsize=11.5,
    fontweight='bold',
    pad=12,
)
ax_cm.set_xlabel('Predicted Land Cover Class', fontsize=10.5, fontweight='bold')
ax_cm.set_ylabel('True Reference Class', fontsize=10.5, fontweight='bold')
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('fig1_confusion_matrix.png', dpi=180)
plt.show()

# ==============================================================================
# WYKRES 2: Obraz w barwach rzeczywistych (RGB: 4-3-2) z nałożonymi etykietami
# ==============================================================================
fig_tc, ax_tc = plt.subplots(figsize=(10, 8.2), dpi=180)
ax_tc.imshow(rgb_img, extent=extent, origin='upper', aspect='equal')

for cls in classes:
  sub = lbl[lbl['label'] == cls]
  ax_tc.scatter(
      sub['x'],
      sub['y'],
      c=class_styles[cls]['color'],
      marker=class_styles[cls]['marker'],
      s=52,
      edgecolors='black',
      linewidths=0.75,
      label=f'{cls} (n={len(sub)})',
      zorder=4,
  )

ax_tc.set_title(
    'True-Color Satellite Composite (R: Band 4, G: Band 3, B: Band 2)\nwith'
    ' Ground-Truth Reference Labels (n = 319)',
    fontsize=11.5,
    fontweight='bold',
    pad=10,
)
ax_tc.set_xlabel('Easting X (m)', fontsize=10)
ax_tc.set_ylabel('Northing Y (m)', fontsize=10)
ax_tc.ticklabel_format(style='plain', useOffset=False)
ax_tc.legend(
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    frameon=True,
    facecolor='white',
    framealpha=0.92,
    title='Land Cover Class',
    title_fontsize=9.5,
    fontsize=8.8,
)
plt.tight_layout()
plt.savefig('fig2_true_color_labels.png', dpi=180, bbox_inches='tight')
plt.show()

# ==============================================================================
# WYKRES 3: Mapa predykcji pokrycia terenu dla wszystkich 180 947 pikseli
# ==============================================================================
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
pred_grid = sat_sorted['pred_label'].map(class_to_idx).values.reshape(ny, nx)

cmap = ListedColormap([class_styles[cls]['color'] for cls in classes])
norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5, 1), len(classes))

fig_map, ax_map = plt.subplots(figsize=(10, 8.2), dpi=180)
ax_map.imshow(
    pred_grid,
    cmap=cmap,
    norm=norm,
    extent=extent,
    origin='upper',
    aspect='equal',
    interpolation='nearest',
)

legend_patches = [
    mpatches.Patch(
        facecolor=class_styles[cls]['color'],
        edgecolor='black',
        linewidth=0.6,
        label=f"{cls} ({(sat_sorted['pred_label']==cls).mean()*100:.1f}%)",
    )
    for cls in classes
]

ax_map.set_title(
    'Predicted Land Cover Classification Map (Random Forest, 180,947 Pixels)',
    fontsize=11.5,
    fontweight='bold',
    pad=10,
)
ax_map.set_xlabel('Easting X (m)', fontsize=10)
ax_map.set_ylabel('Northing Y (m)', fontsize=10)
ax_map.ticklabel_format(style='plain', useOffset=False)
ax_map.legend(
    handles=legend_patches,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    frameon=True,
    facecolor='white',
    framealpha=0.92,
    title='Predicted Class (% Area)',
    title_fontsize=9.5,
    fontsize=8.8,
)
plt.tight_layout()
plt.savefig('fig3_predicted_land_cover_map.png', dpi=180, bbox_inches='tight')
plt.show()