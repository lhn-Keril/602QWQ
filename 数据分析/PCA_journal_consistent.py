# -*- coding: utf-8 -*-
"""
Solvent PCA analysis with journal-ready visualization.

Design goals
------------
1. Keep the original weighted-PCA logic unchanged as much as possible.
2. Upgrade all figures to a publication-ready style.
3. Use the same core color system as the previous modeling script:
   - primary blue   : #1f4e79
   - secondary gray : #7f8c8d
   - accent red     : #c0392b
   - green          : #2e8b57
   - purple         : #6c5ce7
   - gold           : #b8860b
4. Export high-resolution PNG/PDF/TIFF files.
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ========================= Global configuration =========================
OUTPUT_PATH = r'E:\桌面\数据分析'
INPUT_PATH = os.path.join(OUTPUT_PATH, 'data_fixed.xlsx')
SHEET_NAME = 'Sheet1'

FIG_DIR = os.path.join(OUTPUT_PATH, 'figures_journal')
TABLE_DIR = os.path.join(OUTPUT_PATH, 'tables_journal')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

EXPORT_TIFF = False  # Set to True if a journal specifically requests TIFF output.

JOURNAL_COLORS = {
    'primary': '#1f4e79',
    'secondary': '#7f8c8d',
    'accent': '#c0392b',
    'green': '#2e8b57',
    'purple': '#6c5ce7',
    'gold': '#b8860b',
    'grid': '#d9d9d9',
    'light_blue': '#dceaf7',
    'light_red': '#f8d7da',
    'light_gray': '#eceff1',
    'dark': '#2c3e50',
}

SOLVENT_PALETTE = [
    JOURNAL_COLORS['primary'],
    JOURNAL_COLORS['accent'],
    JOURNAL_COLORS['green'],
    JOURNAL_COLORS['purple'],
    JOURNAL_COLORS['gold'],
    '#4f6d7a',
    '#6b8e23',
    '#a26769',
    '#3b6ea5',
    '#8c564b',
    '#5c677d',
    JOURNAL_COLORS['secondary'],
]

JOURNAL_DIVERGING = LinearSegmentedColormap.from_list(
    'journal_diverging',
    [JOURNAL_COLORS['primary'], '#f7f7f7', JOURNAL_COLORS['accent']],
    N=256
)

PROPERTY_NAME_MAP = {
    '分子量': 'Molecular weight',
    '密度_g_ml': 'Density (g mL$^{-1}$)',
    '熔点_C': 'Melting point (°C)',
    '沸点_C': 'Boiling point (°C)',
    '介电常数': 'Dielectric constant',
    '偶极矩_D': 'Dipole moment (D)',
    '极性参数_ETN': 'Polarity parameter $E_T^N$',
    '黏度_cP': 'Viscosity (cP)',
    '表面张力_dyn_cm': 'Surface tension (dyn cm$^{-1}$)',
    '溶解度参数_Hildebrand': 'Hildebrand solubility',
    '氢键供体': 'H-bond donor',
    '氢键受体': 'H-bond acceptor',
    '配位能力': 'Coordination ability',
    '疏水参数_logP': 'Hydrophobicity logP',
    '折射率': 'Refractive index',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'font.weight': 'bold',

    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.4,

    'axes.unicode_minus': False,

    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 4,
    'ytick.major.size': 4,

    'legend.fontsize': 8,
    'legend.frameon': False,

    'figure.dpi': 200,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})
sns.set_style('white')

def style_axes(ax, grid_axis='y', add_grid=False):
    """
    Journal-style axes:
    1. Remove dashed background grid lines.
    2. Add top X-axis and right Y-axis border lines.
    3. Do not add ticks on the top and right axes.
    4. Bold axes, tick labels, and other axis information.
    """
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.4)
        ax.spines[spine].set_color('black')

    ax.grid(False)

    ax.tick_params(
        axis='both',
        which='both',
        top=False,
        right=False,
        width=1.2,
        length=4,
        labelsize=9
    )

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')

    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.title.set_fontweight('bold')

def bold_legend(legend):
    """Bold legend title and labels."""
    if legend is None:
        return

    if legend.get_title() is not None:
        legend.get_title().set_fontweight('bold')

    for text in legend.get_texts():
        text.set_fontweight('bold')

def save_figure(fig, filename_base, dpi=600):
    out_png = os.path.join(FIG_DIR, filename_base + '.png')
    out_pdf = os.path.join(FIG_DIR, filename_base + '.pdf')
    out_tiff = os.path.join(FIG_DIR, filename_base + '.tiff')
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, facecolor='white')
    fig.savefig(out_pdf, facecolor='white')
    if EXPORT_TIFF:
        try:
            fig.savefig(out_tiff, dpi=min(dpi, 300), facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
        except Exception:
            try:
                fig.savefig(out_tiff, dpi=min(dpi, 300), facecolor='white')
            except Exception:
                pass
    plt.close(fig)

# Confidence ellipse helper
# Reference idea adapted from standard covariance ellipse plotting.
def draw_confidence_ellipse(x, y, ax, edgecolor, facecolor='none', n_std=1.7, alpha=0.12, linewidth=1.0):
    if len(x) < 3 or len(y) < 3:
        return
    cov = np.cov(x, y)
    if np.any(~np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if np.any(vals < 0):
        return
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(ellipse)

# ==================== Read data ====================
df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
print(f"Original data shape: {df.shape}")

# ==================== Solvent property database ====================
solvent_properties = {
    'Water': {
        '分子量': 18.02, '密度_g_ml': 1.00, '熔点_C': 0, '沸点_C': 100,
        '介电常数': 80.10, '偶极矩_D': 1.85, '极性参数_ETN': 1.00, '黏度_cP': 1.00,
        '表面张力_dyn_cm': 72.80, '溶解度参数_Hildebrand': 47.90,
        '氢键供体': 1, '氢键受体': 2, '配位能力': 0.5, '疏水参数_logP': -1.38, '折射率': 1.333
    },
    'Ethanol': {
        '分子量': 46.07, '密度_g_ml': 0.789, '熔点_C': -114.1, '沸点_C': 78.37,
        '介电常数': 24.55, '偶极矩_D': 1.69, '极性参数_ETN': 0.654, '黏度_cP': 1.20,
        '表面张力_dyn_cm': 22.27, '溶解度参数_Hildebrand': 26.50,
        '氢键供体': 1, '氢键受体': 1, '配位能力': 0.3, '疏水参数_logP': -0.31, '折射率': 1.361
    },
    'Methanol': {
        '分子量': 32.04, '密度_g_ml': 0.792, '熔点_C': -97.6, '沸点_C': 64.7,
        '介电常数': 32.70, '偶极矩_D': 1.70, '极性参数_ETN': 0.762, '黏度_cP': 0.55,
        '表面张力_dyn_cm': 22.50, '溶解度参数_Hildebrand': 29.60,
        '氢键供体': 1, '氢键受体': 1, '配位能力': 0.3, '疏水参数_logP': -0.77, '折射率': 1.329
    },
    'DMF': {
        '分子量': 73.09, '密度_g_ml': 0.944, '熔点_C': -61, '沸点_C': 153,
        '介电常数': 38.25, '偶极矩_D': 3.86, '极性参数_ETN': 0.404, '黏度_cP': 0.92,
        '表面张力_dyn_cm': 37.10, '溶解度参数_Hildebrand': 24.80,
        '氢键供体': 0, '氢键受体': 1, '配位能力': 0.8, '疏水参数_logP': -1.01, '折射率': 1.430
    },
    'DMSO': {
        '分子量': 78.13, '密度_g_ml': 1.100, '熔点_C': 18.5, '沸点_C': 189,
        '介电常数': 46.45, '偶极矩_D': 3.96, '极性参数_ETN': 0.444, '黏度_cP': 2.00,
        '表面张力_dyn_cm': 43.00, '溶解度参数_Hildebrand': 26.70,
        '氢键供体': 0, '氢键受体': 1, '配位能力': 0.9, '疏水参数_logP': -1.35, '折射率': 1.479
    },
    'Acetone': {
        '分子量': 58.08, '密度_g_ml': 0.791, '熔点_C': -94.7, '沸点_C': 56.05,
        '介电常数': 20.70, '偶极矩_D': 2.88, '极性参数_ETN': 0.355, '黏度_cP': 0.32,
        '表面张力_dyn_cm': 23.70, '溶解度参数_Hildebrand': 20.00,
        '氢键供体': 0, '氢键受体': 1, '配位能力': 0.4, '疏水参数_logP': -0.24, '折射率': 1.359
    },
    'Acetonitrile': {
        '分子量': 41.05, '密度_g_ml': 0.786, '熔点_C': -45, '沸点_C': 82,
        '介电常数': 37.50, '偶极矩_D': 3.92, '极性参数_ETN': 0.460, '黏度_cP': 0.35,
        '表面张力_dyn_cm': 29.30, '溶解度参数_Hildebrand': 24.40,
        '氢键供体': 0, '氢键受体': 1, '配位能力': 0.6, '疏水参数_logP': -0.34, '折射率': 1.344
    },
    'Acetic_acid': {
        '分子量': 60.05, '密度_g_ml': 1.049, '熔点_C': 16.6, '沸点_C': 118.1,
        '介电常数': 6.20, '偶极矩_D': 1.74, '极性参数_ETN': 0.648, '黏度_cP': 1.22,
        '表面张力_dyn_cm': 27.60, '溶解度参数_Hildebrand': 21.40,
        '氢键供体': 1, '氢键受体': 2, '配位能力': 0.3, '疏水参数_logP': -0.17, '折射率': 1.372
    },
    'Ethyl_acetate': {
        '分子量': 88.11, '密度_g_ml': 0.902, '熔点_C': -83.6, '沸点_C': 77.1,
        '介电常数': 6.02, '偶极矩_D': 1.78, '极性参数_ETN': 0.228, '黏度_cP': 0.45,
        '表面张力_dyn_cm': 23.90, '溶解度参数_Hildebrand': 18.60,
        '氢键供体': 0, '氢键受体': 2, '配位能力': 0.2, '疏水参数_logP': 0.73, '折射率': 1.372
    },
    'Pyridine': {
        '分子量': 79.10, '密度_g_ml': 0.983, '熔点_C': -41.6, '沸点_C': 115.2,
        '介电常数': 12.40, '偶极矩_D': 2.22, '极性参数_ETN': 0.302, '黏度_cP': 0.94,
        '表面张力_dyn_cm': 38.00, '溶解度参数_Hildebrand': 21.80,
        '氢键供体': 0, '氢键受体': 1, '配位能力': 0.7, '疏水参数_logP': 0.65, '折射率': 1.509
    },
    'THF': {
        '分子量': 72.11, '密度_g_ml': 0.889, '熔点_C': -108.4, '沸点_C': 66,
        '介电常数': 7.58, '偶极矩_D': 1.75, '极性参数_ETN': 0.207, '黏度_cP': 0.55,
        '表面张力_dyn_cm': 26.40, '溶解度参数_Hildebrand': 19.50,
        '氢键供体': 0, '氢键受体': 1, '配位能力': 0.6, '疏水参数_logP': 0.46, '折射率': 1.407
    }
}

def get_solvent_properties(solvent_name):
    """Return solvent properties. Mixed solvents are averaged by matched components."""
    solvent_name = str(solvent_name)

    if '/' in solvent_name:
        parts = solvent_name.split('/')
        total_props = {}
        count = 0
        for part in parts:
            part = part.strip()
            for key in solvent_properties.keys():
                if key.lower() in part.lower():
                    props = solvent_properties[key]
                    for prop_name, prop_value in props.items():
                        total_props[prop_name] = total_props.get(prop_name, 0) + prop_value
                    count += 1
                    break
        if count > 0:
            return {k: v / count for k, v in total_props.items()}

    for key in solvent_properties.keys():
        if key.lower() in solvent_name.lower():
            return solvent_properties[key].copy()

    return solvent_properties['Water'].copy()

# ==================== Extract solvent parameters ====================
solvent_params_list = []
solvent_names = []

for _, row in df.iterrows():
    solvent = row['Solvent']
    if pd.isna(solvent) or solvent == '':
        solvent = 'Unknown'

    props = get_solvent_properties(solvent)
    solvent_params_list.append(props)

    main_solvent = 'Other'
    for key in solvent_properties.keys():
        if key.lower() in str(solvent).lower():
            main_solvent = key
            break
    solvent_names.append(main_solvent)

solvent_params_df = pd.DataFrame(solvent_params_list)
print(f"Solvent parameter matrix shape: {solvent_params_df.shape}")

pca_features = [
    '介电常数', '偶极矩_D', '极性参数_ETN', '黏度_cP',
    '表面张力_dyn_cm', '溶解度参数_Hildebrand', '氢键供体',
    '氢键受体', '配位能力', '疏水参数_logP', '沸点_C',
    '密度_g_ml', '分子量', '熔点_C', '折射率'
]
available_features = [f for f in pca_features if f in solvent_params_df.columns]
print(f"PCA features ({len(available_features)}): {available_features}")

X = solvent_params_df[available_features].copy()
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# ==================== Weighted PCA ====================
initial_weights = {
    '介电常数': 1.5, '偶极矩_D': 1.3, '极性参数_ETN': 1.8, '黏度_cP': 0.8,
    '表面张力_dyn_cm': 0.7, '溶解度参数_Hildebrand': 1.0, '氢键供体': 1.4,
    '氢键受体': 1.2, '配位能力': 1.6, '疏水参数_logP': 0.9, '沸点_C': 0.6,
    '密度_g_ml': 0.5, '分子量': 0.4, '熔点_C': 0.3, '折射率': 0.5
}

weights = np.array([initial_weights[f] for f in available_features])
X_weighted = X_imputed * weights
X_scaled = StandardScaler().fit_transform(X_weighted)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

if cumulative_variance[1] < 0.85:
    print('PC1 + PC2 below 85%; optimizing weights...')
    from scipy.optimize import minimize

    def objective(weight_params):
        weights_opt = np.maximum(weight_params, 0.1)
        X_weighted_opt = X_imputed * weights_opt
        X_scaled_opt = StandardScaler().fit_transform(X_weighted_opt)
        pca_opt = PCA().fit(X_scaled_opt)
        cum_var = np.cumsum(pca_opt.explained_variance_ratio_)
        return abs(cum_var[1] - 0.85)

    initial_params = np.array([initial_weights[f] for f in available_features])
    result = minimize(objective, initial_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 0.01})
    weights_used = np.maximum(result.x, 0.1)

    X_weighted_opt = X_imputed * weights_used
    X_scaled_opt = StandardScaler().fit_transform(X_weighted_opt)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled_opt)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
else:
    weights_used = np.array([initial_weights[f] for f in available_features])

print(f"PC1 explained variance: {explained_variance_ratio[0] * 100:.2f}%")
print(f"PC2 explained variance: {explained_variance_ratio[1] * 100:.2f}%")
print(f"PC1+PC2 cumulative variance: {cumulative_variance[1] * 100:.2f}%")

# ==================== Add PCA results to original table ====================
df['Solvent_PC1'] = X_pca[:, 0]
df['Solvent_PC2'] = X_pca[:, 1]
if X_pca.shape[1] > 2:
    df['Solvent_PC3'] = X_pca[:, 2]

df['Solvent_Main'] = solvent_names

# ==================== Save tabular outputs ====================
pca_plot_data = pd.DataFrame({
    'Name': df['Name'],
    'Solvent_Original': df['Solvent'],
    'Solvent_Main': df['Solvent_Main'],
    'PC1': df['Solvent_PC1'],
    'PC2': df['Solvent_PC2'],
    'PC3': df['Solvent_PC3'] if 'Solvent_PC3' in df.columns else np.nan,
    'Quantum_yield(%)': df['Quantum yield(%)'],
    'Ex(nm)': df['Ex(nm)'],
    'Em(nm)': df['Em(nm)'],
    'particle_size(nm)': df['particle size(nm)'],
    'stoke(nm)': df['stoke (nm)']
})
pca_plot_data.to_excel(os.path.join(TABLE_DIR, 'solvent_PCA_plot_data.xlsx'), index=False)

variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(min(10, len(explained_variance_ratio)))],
    'Explained Variance Ratio': explained_variance_ratio[:10],
    'Explained Variance (%)': explained_variance_ratio[:10] * 100,
    'Cumulative Variance Ratio': cumulative_variance[:10],
    'Cumulative Variance (%)': cumulative_variance[:10] * 100,
})
variance_df.to_excel(os.path.join(TABLE_DIR, 'solvent_PCA_variance.xlsx'), index=False)

loadings_df = pd.DataFrame(
    pca.components_[:min(5, pca.components_.shape[0]), :].T,
    columns=[f'PC{i+1}' for i in range(min(5, pca.components_.shape[0]))],
    index=[PROPERTY_NAME_MAP.get(f, f) for f in available_features]
)
loadings_df.to_excel(os.path.join(TABLE_DIR, 'solvent_PCA_loadings.xlsx'))

weights_df = pd.DataFrame({
    'Property': [PROPERTY_NAME_MAP.get(f, f) for f in available_features],
    'Weight': weights_used
})
weights_df.to_excel(os.path.join(TABLE_DIR, 'solvent_PCA_feature_weights.xlsx'), index=False)

# Keep the original-style export name too for downstream compatibility.
df.to_excel(os.path.join(OUTPUT_PATH, 'data_fixed_with_PCA.xlsx'), index=False)

# ==================== Publication-ready figures ====================
unique_solvents = list(pd.Series(df['Solvent_Main']).dropna().unique())
color_map = {solv: SOLVENT_PALETTE[i % len(SOLVENT_PALETTE)] for i, solv in enumerate(unique_solvents)}

pc_labels = [f'PC{i+1}' for i in range(len(explained_variance_ratio))]
prop_labels = [PROPERTY_NAME_MAP.get(f, f) for f in available_features]
loadings_pc1_pc2 = pca.components_[:2, :].T
loading_strength = np.sqrt(np.sum(loadings_pc1_pc2 ** 2, axis=1))

# Figure 1: Scree plot
fig, ax = plt.subplots(figsize=(6.2, 4.5))
x = np.arange(1, len(explained_variance_ratio) + 1)
bar_colors = [JOURNAL_COLORS['primary'] if i < 2 else JOURNAL_COLORS['secondary'] for i in range(len(x))]
ax.bar(x, explained_variance_ratio, color=bar_colors, edgecolor='black', linewidth=0.6, alpha=0.88)
ax.plot(x, cumulative_variance, color=JOURNAL_COLORS['accent'], marker='o', linewidth=1.8, markersize=4.8)
ax.axhline(0.85, color=JOURNAL_COLORS['green'], linestyle='--', linewidth=1.2)
ax.text(0.98, 0.87, '85% threshold', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, color=JOURNAL_COLORS['green'], fontweight='bold')
ax.set_xlabel('Principal component')
ax.set_ylabel('Explained variance ratio')
ax.set_title(f'Scree plot of weighted PCA (PC1 + PC2 = {cumulative_variance[1] * 100:.1f}%)')
style_axes(ax, grid_axis='y', add_grid=False)
ax.set_xticks(x)
save_figure(fig, 'Figure_1_scree_plot')

# Figure 2: Variance contribution plot
fig, ax = plt.subplots(figsize=(6.6, 4.5))
show_n = min(10, len(explained_variance_ratio))
x2 = np.arange(1, show_n + 1)
bars = ax.bar(x2, explained_variance_ratio[:show_n] * 100, color=JOURNAL_COLORS['light_blue'],
              edgecolor=JOURNAL_COLORS['primary'], linewidth=0.9)
ax2 = ax.twinx()
ax2.plot(x2, cumulative_variance[:show_n] * 100, color=JOURNAL_COLORS['accent'], marker='o', linewidth=1.8, markersize=4.5)
ax2.axhline(85, color=JOURNAL_COLORS['green'], linestyle='--', linewidth=1.2)
for bar, val in zip(bars, explained_variance_ratio[:show_n] * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f'{val:.1f}',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xlabel('Principal component')
ax.set_ylabel('Explained variance (%)')
ax2.set_ylabel('Cumulative variance (%)')
ax.set_title('Explained and cumulative variance of weighted PCA')
style_axes(ax, grid_axis='y', add_grid=False)

ax2.grid(False)
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax2.spines['top'].set_linewidth(1.4)
ax2.spines['right'].set_linewidth(1.4)
ax2.spines['top'].set_color('black')
ax2.spines['right'].set_color('black')

ax2.tick_params(axis='y', width=1.2, length=4, labelsize=9)
for tick_label in ax2.get_yticklabels():
    tick_label.set_fontweight('bold')
ax2.yaxis.label.set_fontweight('bold')

ax.set_xticks(x2)
save_figure(fig, 'Figure_2_variance_contribution')

# Figure 3: PC1 vs PC2 score plot
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for solv in unique_solvents:
    mask = df['Solvent_Main'] == solv
    if mask.sum() == 0:
        continue
    xvals = df.loc[mask, 'Solvent_PC1'].values
    yvals = df.loc[mask, 'Solvent_PC2'].values
    color = color_map[solv]
    ax.scatter(xvals, yvals, s=42, alpha=0.82, color=color, edgecolor='white', linewidth=0.45, label=solv)
    draw_confidence_ellipse(xvals, yvals, ax=ax, edgecolor=color, facecolor=color)
ax.axhline(0, color=JOURNAL_COLORS['secondary'], linewidth=0.9, alpha=0.7)
ax.axvline(0, color=JOURNAL_COLORS['secondary'], linewidth=0.9, alpha=0.7)
ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.1f}%)')
ax.set_title('Score plot of solvents in PCA space')
style_axes(ax, grid_axis='both', add_grid=False)
leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, title='Solvent', title_fontsize=8)
bold_legend(leg)
save_figure(fig, 'Figure_3_score_plot_PC1_PC2')

# Figure 4: Loading plot
fig, ax = plt.subplots(figsize=(7.0, 7.0))
ax.axhline(0, color=JOURNAL_COLORS['secondary'], linewidth=0.9, alpha=0.7)
ax.axvline(0, color=JOURNAL_COLORS['secondary'], linewidth=0.9, alpha=0.7)
unit_circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', linewidth=1.0,
                         edgecolor=JOURNAL_COLORS['secondary'], alpha=0.55)
ax.add_patch(unit_circle)
for i, (xv, yv) in enumerate(loadings_pc1_pc2):
    mag = loading_strength[i]
    color = JOURNAL_COLORS['accent'] if mag >= np.quantile(loading_strength, 0.67) else JOURNAL_COLORS['primary']
    ax.arrow(0, 0, xv * 0.92, yv * 0.92,
             head_width=0.028, head_length=0.035,
             fc=color, ec=color, linewidth=1.05,
             length_includes_head=True, alpha=0.9)
    ax.text(xv * 1.03, yv * 1.03, prop_labels[i], fontsize=8.2, ha='center', va='center',
            color=JOURNAL_COLORS['dark'], fontweight='bold')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_xlabel(f'PC1 loading ({explained_variance_ratio[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 loading ({explained_variance_ratio[1] * 100:.1f}%)')
ax.set_title('Loading plot of solvent physicochemical properties')
style_axes(ax, grid_axis='both', add_grid=False)
save_figure(fig, 'Figure_4_loading_plot')

# Figure 5: Loadings heatmap
fig, ax = plt.subplots(figsize=(8.0, 6.8))
heatmap_df = pd.DataFrame(
    pca.components_[:min(8, pca.components_.shape[0]), :].T,
    columns=[f'PC{i+1}' for i in range(min(8, pca.components_.shape[0]))],
    index=prop_labels
)
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt='.2f',
    cmap=JOURNAL_DIVERGING,
    center=0,
    linewidths=0.45,
    linecolor='white',
    cbar_kws={'shrink': 0.82, 'label': 'Loading value'},
    annot_kws={'size': 7.6, 'weight': 'bold'},
    ax=ax,
)
ax.set_title('Heatmap of PCA loadings', fontweight='bold')
ax.set_xlabel('Principal component', fontweight='bold')
ax.set_ylabel('Physicochemical property', fontweight='bold')

plt.xticks(rotation=0, fontweight='bold')
plt.yticks(rotation=0, fontweight='bold')

ax.grid(False)
for spine in ['left', 'bottom', 'top', 'right']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.4)
    ax.spines[spine].set_color('black')

ax.tick_params(top=False, right=False, width=1.2, length=4)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_fontweight('bold')
for tick_label in cbar.ax.get_yticklabels():
    tick_label.set_fontweight('bold')

save_figure(fig, 'Figure_5_loadings_heatmap')

# Figure 6: Cumulative variance curve
fig, ax = plt.subplots(figsize=(6.3, 4.5))
ax.fill_between(x, cumulative_variance * 100, color=JOURNAL_COLORS['light_blue'], alpha=0.85)
ax.plot(x, cumulative_variance * 100, color=JOURNAL_COLORS['primary'], marker='o', linewidth=1.9, markersize=4.8)
ax.axhline(85, color=JOURNAL_COLORS['green'], linestyle='--', linewidth=1.2)
ax.axhline(90, color=JOURNAL_COLORS['accent'], linestyle=':', linewidth=1.2)
n85 = np.argmax(cumulative_variance >= 0.85) + 1 if np.any(cumulative_variance >= 0.85) else len(cumulative_variance)
ax.axvline(n85, color=JOURNAL_COLORS['green'], linestyle='--', linewidth=1.0, alpha=0.85)
ax.text(n85 + 0.15, 10, f'{n85} PCs\nreach 85%', color=JOURNAL_COLORS['green'],
        fontsize=8, fontweight='bold')
ax.set_xlabel('Number of principal components')
ax.set_ylabel('Cumulative variance explained (%)')
ax.set_title('Cumulative variance profile of weighted PCA')
style_axes(ax, grid_axis='y', add_grid=False)
ax.set_xticks(x)
save_figure(fig, 'Figure_6_cumulative_variance')

# Figure 7: Feature weight plot
fig, ax = plt.subplots(figsize=(7.4, 5.8))
sorted_idx = np.argsort(weights_used)
sorted_features = [prop_labels[i] for i in sorted_idx]
sorted_weights = weights_used[sorted_idx]
colors_h = [JOURNAL_COLORS['light_blue'] if w < np.median(sorted_weights) else JOURNAL_COLORS['primary'] for w in sorted_weights]
bars = ax.barh(range(len(sorted_features)), sorted_weights, color=colors_h,
               edgecolor=JOURNAL_COLORS['primary'], linewidth=0.75)
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features, fontweight='bold')
ax.set_xlabel('Feature weight')
ax.set_title('Assigned feature weights in weighted PCA')
style_axes(ax, grid_axis='x', add_grid=False)
for bar, val in zip(bars, sorted_weights):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f'{val:.2f}',
            va='center', fontsize=7.8, fontweight='bold')
save_figure(fig, 'Figure_7_feature_weights')

# Figure 8: Eigenvalue plot
fig, ax = plt.subplots(figsize=(6.2, 4.4))
eigenvalues = pca.explained_variance_
eig_x = np.arange(1, len(eigenvalues) + 1)
ax.bar(eig_x, eigenvalues, color=JOURNAL_COLORS['light_red'], edgecolor=JOURNAL_COLORS['accent'], linewidth=0.8)
ax.axhline(1, color=JOURNAL_COLORS['accent'], linestyle='--', linewidth=1.2)
ax.text(0.98, 0.95, 'Kaiser criterion (eigenvalue = 1)', transform=ax.transAxes,
        ha='right', va='top', fontsize=8, color=JOURNAL_COLORS['accent'], fontweight='bold')
ax.set_xlabel('Principal component')
ax.set_ylabel('Eigenvalue')
ax.set_title('Eigenvalue distribution of weighted PCA')
style_axes(ax, grid_axis='y', add_grid=False)
ax.set_xticks(eig_x)
save_figure(fig, 'Figure_8_eigenvalues')

# Figure 9: 3D score plot
if X_pca.shape[1] >= 3:
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, projection='3d')
    for solv in unique_solvents:
        mask = df['Solvent_Main'] == solv
        color = color_map[solv]
        ax.scatter(df.loc[mask, 'Solvent_PC1'], df.loc[mask, 'Solvent_PC2'], df.loc[mask, 'Solvent_PC3'],
                   s=34, alpha=0.84, color=color, edgecolor='white', linewidth=0.35, label=solv)

    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.1f}%)', labelpad=7, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.1f}%)', labelpad=7, fontweight='bold')
    ax.set_zlabel(f'PC3 ({explained_variance_ratio[2] * 100:.1f}%)', labelpad=7, fontweight='bold')
    ax.set_title('3D PCA score plot of solvents', pad=14, fontweight='bold')

    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))

    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["axisline"]["linewidth"] = 1.4

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick_label.set_fontweight('bold')

    leg = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=7.5)
    bold_legend(leg)

    save_figure(fig, 'Figure_9_score_plot_3D')

# Figure 10: PCA biplot
fig, ax = plt.subplots(figsize=(7.3, 5.8))
score_scale_x = df['Solvent_PC1'].abs().max()
score_scale_y = df['Solvent_PC2'].abs().max()
ax.scatter(df['Solvent_PC1'], df['Solvent_PC2'], s=26, color=JOURNAL_COLORS['light_blue'],
           edgecolor=JOURNAL_COLORS['primary'], linewidth=0.35, alpha=0.72)
for i, (xv, yv) in enumerate(loadings_pc1_pc2):
    ax.arrow(0, 0, xv * score_scale_x * 0.55, yv * score_scale_y * 0.55,
             head_width=0.12, head_length=0.16,
             fc=JOURNAL_COLORS['accent'], ec=JOURNAL_COLORS['accent'],
             linewidth=0.9, alpha=0.85, length_includes_head=True)
    ax.text(xv * score_scale_x * 0.62, yv * score_scale_y * 0.62,
            prop_labels[i], fontsize=7.6, color=JOURNAL_COLORS['dark'], fontweight='bold')
ax.axhline(0, color=JOURNAL_COLORS['secondary'], linewidth=0.9, alpha=0.7)
ax.axvline(0, color=JOURNAL_COLORS['secondary'], linewidth=0.9, alpha=0.7)
ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.1f}%)')
ax.set_title('Biplot of PCA scores and loadings')
style_axes(ax, grid_axis='both', add_grid=False)
save_figure(fig, 'Figure_10_biplot')

print('=' * 70)
print(f"Final PC1 + PC2 cumulative variance: {cumulative_variance[1] * 100:.2f}%")
print(f"Figures saved to: {FIG_DIR}")
print(f"Tables saved to: {TABLE_DIR}")
print(f"Data with PCA saved to: {os.path.join(OUTPUT_PATH, 'data_fixed_with_PCA.xlsx')}")
print('=' * 70)