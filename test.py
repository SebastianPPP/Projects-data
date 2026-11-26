import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# 1. Wczytanie danych
df = pd.read_csv('data.csv')

# 2. Konfiguracja stylu i palety kolorów
plt.style.use('seaborn-v0_8-whitegrid')
palette = {'group1': '#1f77b4', 'group2': '#ff7f0e'}

# ==============================================================================
# WYKRES 1: Przegląd danych (trajektoria czasowa, korelacje i surowe relacje)
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 14), dpi=150)
plt.subplots_adjust(hspace=0.35, wspace=0.22)

# Panel 1: dependent vs time
ax = axes[0, 0]
np.random.seed(42)
for grp, sub in df.groupby('group'):
  jitter = np.random.uniform(-0.12, 0.12, size=len(sub))
  ax.scatter(
      sub['time'] + jitter,
      sub['dependent'],
      alpha=0.5,
      s=24,
      color=palette[grp],
      label=f'{grp} points',
  )
  means = sub.groupby('time')['dependent'].mean()
  ax.plot(
      means.index,
      means.values,
      color=palette[grp],
      linewidth=2.5,
      marker='o',
      markersize=5,
      label=f'{grp} mean',
  )
ax.set_title(
    '1. Dependent vs. Time (Nonlinear vs. Linear Growth)',
    fontsize=11,
    fontweight='bold',
)
ax.set_xlabel('time')
ax.set_ylabel('dependent')
ax.set_xticks(range(1, 11))
ax.legend(fontsize=8.5, loc='upper left')

# Panel 2: Heatmapa korelacji Pearsona z dependent w podziale na grupy
ax = axes[0, 1]
corr_g1 = (
    df[df['group'] == 'group1']
    .drop(columns=['group'])
    .corr()['dependent']
    .drop('dependent')
)
corr_g2 = (
    df[df['group'] == 'group2']
    .drop(columns=['group'])
    .corr()['dependent']
    .drop('dependent')
)
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
)
ax.set_title(
    '2. Pearson Correlation with Dependent by Group',
    fontsize=11,
    fontweight='bold',
)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Panele 3-6: Wykresy rozrzutu dla var1, var2, var3, var4 z liniami trendu
vars_to_plot = ['var1', 'var2', 'var3', 'var4']
feature_axes = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
titles = [
    '3. Dependent vs. var1 (Strong positive slope in group2)',
    '4. Dependent vs. var2 (Positive effect in group1, flat in group2)',
    '5. Dependent vs. var3 (Weak / negligible linear trend)',
    '6. Dependent vs. var4 (Strong positive slope in both groups)',
]

for ax, var, title in zip(feature_axes, vars_to_plot, titles):
  for grp, sub in df.groupby('group'):
    ax.scatter(
        sub[var],
        sub['dependent'],
        alpha=0.5,
        s=24,
        color=palette[grp],
        label=grp,
    )
    m, b = np.polyfit(sub[var], sub['dependent'], 1)
    x_vals = np.linspace(sub[var].min(), sub[var].max(), 100)
    ax.plot(
        x_vals, m * x_vals + b, color=palette[grp], linewidth=2, linestyle='--'
    )
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_xlabel(var)
  ax.set_ylabel('dependent')
  ax.legend(fontsize=9)

fig.suptitle(
    'Exploratory Data Visualization: Group Regimes & Predictor Relationships (n'
    ' = 400)',
    fontsize=14,
    fontweight='bold',
    y=0.995,
)
plt.savefig('eda_overview.png', bbox_inches='tight')
plt.show()

# ==============================================================================
# WYKRES 2: Efekty cząstkowe po usunięciu grupowych trendów czasowych (reszty)
# ==============================================================================
df_res = df.copy()
for grp in ['group1', 'group2']:
  mask = df_res['group'] == grp
  time_model = smf.ols(
      'dependent ~ time + I(time**2) + I(time**3)', data=df_res[mask]
  ).fit()
  df_res.loc[mask, 'dep_detrended'] = time_model.resid

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=150)
plt.subplots_adjust(wspace=0.28)

for ax, var in zip(axes, vars_to_plot):
  for grp, sub in df_res.groupby('group'):
    ax.scatter(
        sub[var],
        sub['dep_detrended'],
        alpha=0.55,
        s=24,
        color=palette[grp],
        label=grp,
    )
    m, b = np.polyfit(sub[var], sub['dep_detrended'], 1)
    x_vals = np.linspace(sub[var].min(), sub[var].max(), 100)
    ax.plot(
        x_vals, m * x_vals + b, color=palette[grp], linewidth=2, linestyle='--'
    )
  ax.axhline(0, color='gray', linestyle=':', linewidth=1)
  ax.set_title(
      f'Detrended Dependent vs. {var}', fontsize=10.5, fontweight='bold'
  )
  ax.set_xlabel(var)
  if var == 'var1':
    ax.set_ylabel('Residual (after removing time trend)')
  ax.legend(fontsize=8.5)

fig.suptitle(
    'Partial Predictor Effects After Removing Group-Specific Time Trends',
    fontsize=12.5,
    fontweight='bold',
    y=1.03,
)
plt.savefig('eda_detrended.png', bbox_inches='tight')
plt.show()