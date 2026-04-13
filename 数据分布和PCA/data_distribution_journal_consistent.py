import os
import tempfile
from pathlib import Path
import warnings

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mplconfig_journal'))

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, norm

warnings.filterwarnings('ignore')

EXPORT_PDF = True
EXPORT_TIFF = False
PNG_DPI = 500
TIFF_DPI = 600

PALETTE = {
    'blue': '#2F5D8C',
    'gray': '#9AA5B1',
    'red': '#C44E52',
    'green': '#55A868',
    'purple': '#8172B3',
    'gold': '#CCB974',
    'light_blue': '#B8CDE6',
    'light_gray': '#E3E7EB',
    'black': '#2F2F2F',
    'white': '#FFFFFF',
    'soft_red': '#EBC3C4',
    'soft_green': '#CFE8D4',
    'soft_purple': '#D8D1EA',
    'soft_gold': '#E9DFBC',
}

ACCENT_COLORS = [
    PALETTE['blue'],
    PALETTE['gray'],
    PALETTE['green'],
    PALETTE['purple'],
    PALETTE['gold'],
    '#6FA8DC',
    '#B3B3B3',
]

TEMP_COL = 'Temperature (\u2103)'
WORK_DIR = r'E:\桌面\数据分布和相关性分析'

LABEL_MAP = {
    'Quantum yield(%)': 'Quantum yield (%)',
    'Em(nm)': 'Emission wavelength (nm)',
    'stoke (nm)': 'Stokes shift (nm)',
    'particle size(nm)': 'Particle size (nm)',
    TEMP_COL: 'Temperature (degC)',
    'Time (h)': 'Reaction time (h)',
    'volume (mL)': 'Volume (mL)',
    'Ex(nm)': 'Excitation wavelength (nm)',
    'Abs(nm)': 'Absorption wavelength (nm)',
    'Material 1 (M)': 'Material 1 concentration (M)',
    'Material 2 (M)': 'Material 2 concentration (M)',
    'Material 3 (M)': 'Material 3 concentration (M)',
    'Solvent_PC1': 'Solvent PC1',
    'Solvent_PC2': 'Solvent PC2',
}

def configure_publication_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.weight': 'bold',

        'axes.unicode_minus': False,
        'axes.linewidth': 1.5,
        'axes.edgecolor': PALETTE['black'],
        'axes.labelcolor': PALETTE['black'],
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',

        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': PALETTE['black'],
        'ytick.color': PALETTE['black'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,

        'legend.frameon': False,
        'legend.fontsize': 9,

        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
    })
    sns.set_theme(style='white')

def resolve_base_dir():
    preferred = Path(WORK_DIR)
    if preferred.exists():
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    fallback = Path(__file__).resolve().parent / 'data_distribution_journal_outputs'
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

def ensure_dirs(base_dir):
    dirs = {
        'base': base_dir,
        'fig_root': base_dir / 'figures_journal',
        'dist': base_dir / 'figures_journal' / '01_distribution',
        'group': base_dir / 'figures_journal' / '02_group_comparison',
        'corr': base_dir / 'figures_journal' / '03_correlation',
        'qc': base_dir / 'figures_journal' / '04_quality_control',
        'tables': base_dir / 'tables_journal',
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

def style_axes(ax):
    # 四边框
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color(PALETTE['black'])

    # 去掉背景网格
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # 上右只有轴线，不要刻度
    ax.tick_params(
        axis='both',
        which='major',
        length=4,
        width=1.2,
        top=False,
        right=False,
        labelsize=10
    )

    # 刻度文字加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # 坐标轴标题、图题加粗
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.title.set_fontweight('bold')

def style_heatmap_axes(ax):
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color(PALETTE['black'])

    ax.grid(False)
    ax.tick_params(axis='both', which='major', top=False, right=False, width=1.2, length=4)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.title.set_fontweight('bold')

def bold_legend(legend):
    if legend is None:
        return
    if legend.get_title() is not None:
        legend.get_title().set_fontweight('bold')
    for text in legend.get_texts():
        text.set_fontweight('bold')

def export_figure(fig, stem_path):
    stem_path = Path(stem_path)
    fig.savefig(stem_path.with_suffix('.png'), dpi=PNG_DPI, bbox_inches='tight')
    if EXPORT_PDF:
        fig.savefig(stem_path.with_suffix('.pdf'), bbox_inches='tight')
    if EXPORT_TIFF:
        fig.savefig(stem_path.with_suffix('.tiff'), dpi=TIFF_DPI, bbox_inches='tight')
    plt.close(fig)

def save_excel_sheets(path, sheets):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce').dropna()

def pretty_label(name):
    return LABEL_MAP.get(name, name)

def classify_solvent(solvent):
    if pd.isna(solvent):
        return 'Unknown'
    s = str(solvent).strip().lower().replace('\xa0', ' ')
    if 'water' in s:
        return 'Water'
    if 'ethanol' in s:
        return 'Ethanol'
    if 'dmf' in s:
        return 'DMF'
    if 'methanol' in s:
        return 'Methanol'
    if 'acetic' in s:
        return 'Acetic acid'
    return 'Mixed/Other'

def classify_precursor(smiles):
    if pd.isna(smiles):
        return 'None'
    s = str(smiles).lower()
    if 'op(=o)(o)o' in s:
        return 'Phosphorus'
    if 'b(o)o' in s:
        return 'Boron'
    if 's(=o)(=o)(o)o' in s:
        return 'Sulfur'
    if any(x in s for x in ['c', 'n', 'o']):
        return 'Organic'
    return 'Other'

def validate_smiles(smiles):
    if pd.isna(smiles):
        return 'Missing'
    s = str(smiles).strip()
    if len(s) == 0:
        return 'Empty'
    if any(c in s for c in ['C', 'c', 'N', 'n', 'O', 'o']):
        return 'Valid'
    return 'Invalid'

def choose_bins(data, default_bins=28):
    if len(data) < 2:
        return 8
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return default_bins
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    if bin_width <= 0:
        return default_bins
    return int(np.clip(np.ceil((data.max() - data.min()) / bin_width), 8, 40))

def palette_for_categories(categories):
    return {cat: ACCENT_COLORS[i % len(ACCENT_COLORS)] for i, cat in enumerate(categories)}

def plot_histogram_with_fit(data, save_column, x_label, title, stem_name, dirs):
    data = pd.Series(data).dropna().astype(float)
    if len(data) == 0:
        return

    bins = choose_bins(data)
    counts, hist_bins = np.histogram(data, bins=bins)
    centers = (hist_bins[:-1] + hist_bins[1:]) / 2

    raw_df = pd.DataFrame({save_column: data.values, 'Frequency': np.ones(len(data), dtype=int)})
    hist_df = pd.DataFrame({
        'Bin_Start': hist_bins[:-1],
        'Bin_End': hist_bins[1:],
        'Bin_Center': centers,
        'Count': counts,
        'Density': counts / max(len(data), 1),
    })
    save_excel_sheets(dirs['tables'] / f'{stem_name}.xlsx', {
        'Raw_Data': raw_df,
        'Histogram_Data': hist_df,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.hist(
        data,
        bins=bins,
        density=True,
        color=PALETTE['light_blue'],
        edgecolor=PALETTE['blue'],
        linewidth=1.0,
        alpha=0.88,
        label='Histogram'
    )

    mu, sigma = norm.fit(data)
    x = np.linspace(data.min(), data.max(), 400)
    ax.plot(x, norm.pdf(x, mu, sigma), color=PALETTE['red'], linewidth=2.2, label='Normal fit')

    if len(data) > 3 and data.nunique() > 1:
        try:
            kde = gaussian_kde(data)
            ax.plot(x, kde(x), color=PALETTE['green'], linewidth=2.0, linestyle='--', label='KDE')
        except Exception:
            pass

    ax.axvline(mu, color=PALETTE['gold'], linestyle=':', linewidth=1.6, label=f'Mean = {mu:.2f}')

    style_axes(ax)
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title(f'{title} (n = {len(data)})', pad=10, fontweight='bold')

    stat_text = f'Mean = {data.mean():.2f}\nMedian = {data.median():.2f}\nSD = {data.std(ddof=1):.2f}'
    ax.text(
        0.98, 0.98, stat_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=9, fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor='white',
            edgecolor=PALETTE['light_gray'],
            linewidth=1.0
        )
    )

    leg = ax.legend(loc='upper left')
    bold_legend(leg)

    fig.tight_layout()
    export_figure(fig, dirs['dist'] / stem_name)

def plot_group_violin(data_dict, x_name, y_name, y_label, title, stem_name, dirs, order=None):
    rows = []
    for cat, vals in data_dict.items():
        vals = pd.Series(vals).dropna().astype(float)
        for v in vals:
            rows.append({x_name: cat, y_name: v})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    stats_rows = []
    for cat, vals in data_dict.items():
        vals = pd.Series(vals).dropna().astype(float)
        if len(vals) == 0:
            continue
        stats_rows.append({
            'Category': cat,
            'Count': len(vals),
            'Mean': vals.mean(),
            'Median': vals.median(),
            'SD': vals.std(ddof=1),
            'Min': vals.min(),
            'Max': vals.max(),
            'Q1': vals.quantile(0.25),
            'Q3': vals.quantile(0.75),
        })
    save_excel_sheets(dirs['tables'] / f'{stem_name}.xlsx', {
        'Raw_Data': plot_df,
        'Summary_Statistics': pd.DataFrame(stats_rows),
    })

    if order is None:
        order = [cat for cat in data_dict if len(pd.Series(data_dict[cat]).dropna()) > 0]
    else:
        order = [cat for cat in order if cat in data_dict and len(pd.Series(data_dict[cat]).dropna()) > 0]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    sns.violinplot(
        data=plot_df,
        x=x_name,
        y=y_name,
        order=order,
        palette=palette_for_categories(order),
        inner=None,
        cut=0,
        linewidth=1.0,
        saturation=0.95,
        ax=ax
    )

    sns.boxplot(
        data=plot_df,
        x=x_name,
        y=y_name,
        order=order,
        width=0.16,
        showcaps=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor=PALETTE['black'], linewidth=1.1, zorder=3),
        whiskerprops=dict(color=PALETTE['black'], linewidth=1.1),
        capprops=dict(color=PALETTE['black'], linewidth=1.1),
        medianprops=dict(color=PALETTE['red'], linewidth=1.5),
        ax=ax
    )

    sns.stripplot(
        data=plot_df,
        x=x_name,
        y=y_name,
        order=order,
        color=PALETTE['gray'],
        size=2.8,
        jitter=0.18,
        alpha=0.40,
        linewidth=0,
        ax=ax
    )

    style_axes(ax)
    ax.set_xlabel('', fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, pad=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=28)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    fig.tight_layout()
    export_figure(fig, dirs['group'] / stem_name)

def plot_correlation_heatmap(corr_matrix, dirs):
    if corr_matrix.empty:
        return

    save_excel_sheets(dirs['base'] / 'correlation_analysis_tables.xlsx', {
        'Correlation_Matrix': corr_matrix.reset_index().rename(columns={'index': 'Variable'})
    })

    cmap = LinearSegmentedColormap.from_list(
        'journal_diverging',
        [PALETTE['blue'], PALETTE['white'], PALETTE['red']],
        N=256
    )

    fig, ax = plt.subplots(figsize=(10.0, 8.6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        linecolor='white',
        cbar_kws={'shrink': 0.85, 'label': 'Pearson correlation coefficient'},
        annot_kws={'size': 8, 'weight': 'bold'},
        ax=ax
    )

    ax.set_title('Correlation matrix of experimental variables', pad=12, fontweight='bold')
    ax.set_xlabel('', fontweight='bold')
    ax.set_ylabel('', fontweight='bold')

    style_heatmap_axes(ax)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontweight('bold')
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    fig.tight_layout()
    export_figure(fig, dirs['corr'] / 'correlation_heatmap')

def plot_target_correlation_bars(corr_matrix, target, dirs):
    if target not in corr_matrix.columns:
        return

    corr_series = corr_matrix[target].drop(labels=[target], errors='ignore').dropna().sort_values()
    if corr_series.empty:
        return

    safe_target = target.replace('%', 'pct').replace('/', '_').replace(' ', '_')
    export_df = pd.DataFrame({
        'Variable': corr_series.index,
        'Correlation': corr_series.values,
        'Absolute_Correlation': np.abs(corr_series.values),
    }).sort_values('Absolute_Correlation', ascending=False)

    save_excel_sheets(dirs['tables'] / f'correlation_with_{safe_target}.xlsx', {
        'Correlation_Ranking': export_df
    })

    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    colors = [PALETTE['red'] if v < 0 else PALETTE['blue'] for v in corr_series.values]

    ax.barh(
        [pretty_label(v) for v in corr_series.index],
        corr_series.values,
        color=colors,
        edgecolor=PALETTE['black'],
        linewidth=0.7
    )
    ax.axvline(0, color=PALETTE['black'], linewidth=1.1)

    style_axes(ax)
    ax.set_xlabel('Pearson correlation coefficient', fontweight='bold')
    ax.set_ylabel('', fontweight='bold')
    ax.set_title(f'Correlation with {pretty_label(target)}', pad=10, fontweight='bold')

    fig.tight_layout()
    export_figure(fig, dirs['corr'] / f'correlation_with_{safe_target}')

def plot_smiles_validation(smiles_df, dirs):
    if smiles_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    pivot = smiles_df.pivot(index='Column', columns='Status', values='Count').fillna(0)
    order = [col for col in ['Valid', 'Missing', 'Empty', 'Invalid'] if col in pivot.columns]
    pivot = pivot[order]

    status_palette = {
        'Valid': PALETTE['blue'],
        'Missing': PALETTE['gray'],
        'Empty': PALETTE['gold'],
        'Invalid': PALETTE['red']
    }

    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))

    for status in order:
        values = pivot[status].values
        ax.barh(
            y,
            values,
            left=left,
            color=status_palette[status],
            edgecolor='white',
            linewidth=0.6,
            label=status
        )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontweight='bold')
    ax.set_xlabel('Count', fontweight='bold')
    ax.set_ylabel('', fontweight='bold')
    ax.set_title('SMILES validation summary', pad=10, fontweight='bold')

    style_axes(ax)

    leg = ax.legend(loc='lower right', ncol=min(4, len(order)))
    bold_legend(leg)

    fig.tight_layout()
    export_figure(fig, dirs['qc'] / 'smiles_validation_summary')

def write_report(base_dir, dirs, df, corr_vars):
    lines = [
        '=' * 60,
        'Carbon-dot dataset report (journal-style visualization version)',
        '=' * 60,
        '',
        'Dataset summary:',
        f'- Samples: {len(df)}',
        f'- Numeric variables: {len(corr_vars)}',
        '',
        'Output directories:',
        f'- Figures: {dirs["fig_root"]}',
        f'- Tables: {dirs["tables"]}',
        '',
        'Visual upgrades:',
        '- Unified journal-style typography, line widths, axes and spacing',
        '- Consistent palette with the previous two scripts: blue / gray / red / green / purple / gold',
        '- Distribution figures include normal fit and KDE',
        '- Group comparison figures use violin + box + jitter',
        '- Correlation figures use a blue-white-red diverging map',
        '- Background dashed grid removed',
        '- Top X-axis and right Y-axis border lines added without ticks',
        '- Axis and figure text information bolded',
        '- Final version optimized for publication-style aesthetics',
        '',
        'Export formats:',
        f'- PNG ({PNG_DPI} dpi)',
        f'- PDF: {EXPORT_PDF}',
        f'- TIFF: {EXPORT_TIFF}',
    ]
    with open(base_dir / 'journal_visualization_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def main():
    configure_publication_style()
    base_dir = resolve_base_dir()
    os.chdir(base_dir)
    dirs = ensure_dirs(base_dir)

    data_path = base_dir / 'data_fixed_with_PCA.xlsx'
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent / 'data_fixed_with_PCA.xlsx'
    if not data_path.exists():
        raise FileNotFoundError('Could not find data_fixed_with_PCA.xlsx in the working directory or script directory.')

    df = pd.read_excel(data_path, sheet_name='Sheet1')
    df.columns = df.columns.str.strip()

    print('Dataset info: - data_distribution_journal_consistent.py:606')
    print(f'Shape: {df.shape} - data_distribution_journal_consistent.py:607')
    print(f'Columns: {df.columns.tolist()} - data_distribution_journal_consistent.py:608')

    numeric_columns = [
        'Material 1 (M)', 'Material 2 (M)', 'Material 3 (M)',
        'volume (mL)', TEMP_COL, 'Time (h)',
        'Quantum yield(%)', 'Ex(nm)', 'Em(nm)',
        'particle size(nm)', 'stoke (nm)', 'Abs(nm)',
        'Solvent_PC1', 'Solvent_PC2'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Solvent' in df.columns:
        df['Solvent_Type'] = df['Solvent'].apply(classify_solvent)
    else:
        df['Solvent_Type'] = 'Unknown'

    smiles_cols = [c for c in ['Material 1 (SMILES)', 'Material 2 (SMILES)', 'Material 3 (SMILES)'] if c in df.columns]
    df['Material_Count'] = df[smiles_cols].notna().sum(axis=1) if smiles_cols else 0

    if 'Material 1 (SMILES)' in df.columns:
        df['Precursor1_Type'] = df['Material 1 (SMILES)'].apply(classify_precursor)
    else:
        df['Precursor1_Type'] = 'Unknown'

    distribution_specs = [
        ('Quantum yield(%)', 'Quantum_yield', 'Quantum yield (%)', 'Distribution of quantum yield', 'distribution_quantum_yield'),
        ('Em(nm)', 'Emission_wavelength', 'Emission wavelength (nm)', 'Distribution of emission wavelength', 'distribution_emission_wavelength'),
        ('stoke (nm)', 'Stokes_shift', 'Stokes shift (nm)', 'Distribution of Stokes shift', 'distribution_stokes_shift'),
        ('particle size(nm)', 'Particle_size', 'Particle size (nm)', 'Distribution of particle size', 'distribution_particle_size'),
        (TEMP_COL, 'Temperature', 'Temperature (degC)', 'Distribution of reaction temperature', 'distribution_temperature'),
        ('Time (h)', 'Reaction_time', 'Reaction time (h)', 'Distribution of reaction time', 'distribution_time'),
    ]

    for col, save_col, xlabel, title, stem in distribution_specs:
        if col in df.columns:
            data = safe_numeric(df[col])
            if len(data) > 0:
                plot_histogram_with_fit(data, save_col, xlabel, title, stem, dirs)

    solvent_order = ['Water', 'Ethanol', 'DMF', 'Methanol', 'Acetic acid', 'Mixed/Other', 'Unknown']

    if 'Quantum yield(%)' in df.columns:
        solvent_qy = {
            s: safe_numeric(df.loc[df['Solvent_Type'] == s, 'Quantum yield(%)']).values
            for s in df['Solvent_Type'].dropna().unique()
        }
        plot_group_violin(
            solvent_qy, 'Solvent_Type', 'Quantum yield(%)', 'Quantum yield (%)',
            'Quantum yield across solvent types', 'violin_qy_solvent_type', dirs, order=solvent_order
        )

        qy_temp = df[[TEMP_COL, 'Quantum yield(%)']].dropna().copy()
        if not qy_temp.empty:
            qy_temp['Temperature_Group'] = pd.cut(
                qy_temp[TEMP_COL],
                bins=[0, 120, 160, 200, 240, 300],
                labels=['<120', '120-160', '160-200', '200-240', '>240']
            )
            temp_dict = {
                str(g): qy_temp.loc[qy_temp['Temperature_Group'] == g, 'Quantum yield(%)'].values
                for g in qy_temp['Temperature_Group'].dropna().cat.categories
            }
            plot_group_violin(
                temp_dict, 'Temperature_Group', 'Quantum yield(%)', 'Quantum yield (%)',
                'Quantum yield across temperature ranges', 'violin_qy_temperature_range', dirs,
                order=['<120', '120-160', '160-200', '200-240', '>240']
            )

        qy_time = df[['Time (h)', 'Quantum yield(%)']].dropna().copy()
        if not qy_time.empty:
            qy_time['Time_Group'] = pd.cut(
                qy_time['Time (h)'],
                bins=[0, 2, 6, 10, 15, 25, 100],
                labels=['<2', '2-6', '6-10', '10-15', '15-25', '>25']
            )
            time_dict = {
                str(g): qy_time.loc[qy_time['Time_Group'] == g, 'Quantum yield(%)'].values
                for g in qy_time['Time_Group'].dropna().cat.categories
            }
            plot_group_violin(
                time_dict, 'Time_Group', 'Quantum yield(%)', 'Quantum yield (%)',
                'Quantum yield across reaction time ranges', 'violin_qy_reaction_time', dirs,
                order=['<2', '2-6', '6-10', '10-15', '15-25', '>25']
            )

        material_qy = {
            f'{int(count)} materials': safe_numeric(df.loc[df['Material_Count'] == count, 'Quantum yield(%)']).values
            for count in sorted(df['Material_Count'].dropna().unique())
        }
        plot_group_violin(
            material_qy, 'Material_Count', 'Quantum yield(%)', 'Quantum yield (%)',
            'Quantum yield across material counts', 'violin_qy_material_count', dirs
        )

        precursor_qy = {
            p: safe_numeric(df.loc[df['Precursor1_Type'] == p, 'Quantum yield(%)']).values
            for p in df['Precursor1_Type'].dropna().unique()
        }
        plot_group_violin(
            precursor_qy, 'Precursor_Type', 'Quantum yield(%)', 'Quantum yield (%)',
            'Quantum yield across precursor classes', 'violin_qy_precursor_type', dirs
        )

    if 'Em(nm)'in df.columns:
        if 'Ex(nm)' in df.columns:
         ex_solvent = {
            s: safe_numeric(df.loc[df['Solvent_Type'] == s, 'Ex(nm)']).values
            for s in df['Solvent_Type'].dropna().unique()
        }
        plot_group_violin(
            ex_solvent, 'Solvent_Type', 'Ex(nm)', 'Excitation wavelength (nm)',
            'Excitation wavelength across solvent types', 'violin_ex_solvent_type', dirs, order=solvent_order
        )

        ex_temp = df[[TEMP_COL, 'Ex(nm)']].dropna().copy()
        if not ex_temp.empty:
            ex_temp['Temperature_Group'] = pd.cut(
                ex_temp[TEMP_COL],
                bins=[0, 120, 160, 200, 240, 300],
                labels=['<120', '120-160', '160-200', '200-240', '>240']
            )
            temp_dict = {
                str(g): ex_temp.loc[ex_temp['Temperature_Group'] == g, 'Ex(nm)'].values
                for g in ex_temp['Temperature_Group'].dropna().cat.categories
            }
            plot_group_violin(
                temp_dict, 'Temperature_Group', 'Ex(nm)', 'Excitation wavelength (nm)',
                'Excitation wavelength across temperature ranges', 'violin_ex_temperature_range', dirs,
                order=['<120', '120-160', '160-200', '200-240', '>240']
            )

        ex_material = {
            f'{int(count)} materials': safe_numeric(df.loc[df['Material_Count'] == count, 'Ex(nm)']).values
            for count in sorted(df['Material_Count'].dropna().unique())
        }
        plot_group_violin(
            ex_material, 'Material_Count', 'Ex(nm)', 'Excitation wavelength (nm)',
            'Excitation wavelength across material counts', 'violin_ex_material_count', dirs
        )

        ex_time = df[['Time (h)', 'Ex(nm)']].dropna().copy()
        if not ex_time.empty:
            ex_time['Time_Group'] = pd.cut(
                ex_time['Time (h)'],
                bins=[0, 2, 6, 10, 15, 25, 100],
                labels=['<2', '2-6', '6-10', '10-15', '15-25', '>25']
            )
            time_dict = {
                str(g): ex_time.loc[ex_time['Time_Group'] == g, 'Ex(nm)'].values
                for g in ex_time['Time_Group'].dropna().cat.categories
            }
            plot_group_violin(
                time_dict, 'Time_Group', 'Ex(nm)', 'Excitation wavelength (nm)',
                'Excitation wavelength across reaction time ranges', 'violin_ex_reaction_time', dirs,
                order=['<2', '2-6', '6-10', '10-15', '15-25', '>25']
            )

        ex_precursor = {
            p: safe_numeric(df.loc[df['Precursor1_Type'] == p, 'Ex(nm)']).values
            for p in df['Precursor1_Type'].dropna().unique()
        }
        plot_group_violin(
            ex_precursor, 'Precursor_Type', 'Ex(nm)', 'Excitation wavelength (nm)',
            'Excitation wavelength across precursor classes', 'violin_ex_precursor_type', dirs
        )
        solvent_em = {
            s: safe_numeric(df.loc[df['Solvent_Type'] == s, 'Em(nm)']).values
            for s in df['Solvent_Type'].dropna().unique()
        }
        plot_group_violin(
            solvent_em, 'Solvent_Type', 'Em(nm)', 'Emission wavelength (nm)',
            'Emission wavelength across solvent types', 'violin_em_solvent_type', dirs, order=solvent_order
        )

        em_temp = df[[TEMP_COL, 'Em(nm)']].dropna().copy()
        if not em_temp.empty:
            em_temp['Temperature_Group'] = pd.cut(
                em_temp[TEMP_COL],
                bins=[0, 120, 160, 200, 240, 300],
                labels=['<120', '120-160', '160-200', '200-240', '>240']
            )
            temp_dict = {
                str(g): em_temp.loc[em_temp['Temperature_Group'] == g, 'Em(nm)'].values
                for g in em_temp['Temperature_Group'].dropna().cat.categories
            }
            plot_group_violin(
                temp_dict, 'Temperature_Group', 'Em(nm)', 'Emission wavelength (nm)',
                'Emission wavelength across temperature ranges', 'violin_em_temperature_range', dirs,
                order=['<120', '120-160', '160-200', '200-240', '>240']
            )

        material_em = {
            f'{int(count)} materials': safe_numeric(df.loc[df['Material_Count'] == count, 'Em(nm)']).values
            for count in sorted(df['Material_Count'].dropna().unique())
        }
        plot_group_violin(
            material_em, 'Material_Count', 'Em(nm)', 'Emission wavelength (nm)',
            'Emission wavelength across material counts', 'violin_em_material_count', dirs
        )

    if 'stoke (nm)' in df.columns:
        solvent_stokes = {
            s: safe_numeric(df.loc[df['Solvent_Type'] == s, 'stoke (nm)']).values
            for s in df['Solvent_Type'].dropna().unique()
        }
        plot_group_violin(
            solvent_stokes, 'Solvent_Type', 'stoke (nm)', 'Stokes shift (nm)',
            'Stokes shift across solvent types', 'violin_stokes_solvent_type', dirs, order=solvent_order
        )

        stokes_temp = df[[TEMP_COL, 'stoke (nm)']].dropna().copy()
        if not stokes_temp.empty:
            stokes_temp['Temperature_Group'] = pd.cut(
                stokes_temp[TEMP_COL],
                bins=[0, 140, 180, 220, 300],
                labels=['<140', '140-180', '180-220', '>220']
            )
            temp_dict = {
                str(g): stokes_temp.loc[stokes_temp['Temperature_Group'] == g, 'stoke (nm)'].values
                for g in stokes_temp['Temperature_Group'].dropna().cat.categories
            }
            plot_group_violin(
                temp_dict, 'Temperature_Group', 'stoke (nm)', 'Stokes shift (nm)',
                'Stokes shift across temperature ranges', 'violin_stokes_temperature_range', dirs,
                order=['<140', '140-180', '180-220', '>220']
            )

    if 'particle size(nm)' in df.columns:
        solvent_size = {
            s: safe_numeric(df.loc[df['Solvent_Type'] == s, 'particle size(nm)']).values
            for s in df['Solvent_Type'].dropna().unique()
        }
        plot_group_violin(
            solvent_size, 'Solvent_Type', 'particle size(nm)', 'Particle size (nm)',
            'Particle size across solvent types', 'violin_particle_size_solvent_type', dirs, order=solvent_order
        )

        size_temp = df[[TEMP_COL, 'particle size(nm)']].dropna().copy()
        if not size_temp.empty:
            size_temp['Temperature_Group'] = pd.cut(
                size_temp[TEMP_COL],
                bins=[0, 140, 180, 220, 300],
                labels=['<140', '140-180', '180-220', '>220']
            )
            temp_dict = {
                str(g): size_temp.loc[size_temp['Temperature_Group'] == g, 'particle size(nm)'].values
                for g in size_temp['Temperature_Group'].dropna().cat.categories
            }
            plot_group_violin(
                temp_dict, 'Temperature_Group', 'particle size(nm)', 'Particle size (nm)',
                'Particle size across temperature ranges', 'violin_particle_size_temperature_range', dirs,
                order=['<140', '140-180', '180-220', '>220']
            )

    corr_vars = [
        'Quantum yield(%)', 'Em(nm)', 'stoke (nm)', 'particle size(nm)',
        TEMP_COL, 'Time (h)', 'volume (mL)', 'Ex(nm)', 'Abs(nm)',
        'Material 1 (M)', 'Material 2 (M)', 'Solvent_PC1', 'Solvent_PC2'
    ]
    corr_vars = [v for v in corr_vars if v in df.columns]
    corr_matrix = df[corr_vars].corr()

    plot_correlation_heatmap(corr_matrix, dirs)
    for target in ['Quantum yield(%)', 'Em(nm)', 'stoke (nm)']:
        plot_target_correlation_bars(corr_matrix, target, dirs)

    smiles_rows = []
    for col in ['Material 1 (SMILES)', 'Material 2 (SMILES)', 'Material 3 (SMILES)']:
        if col in df.columns:
            df[f'{col}_Status'] = df[col].apply(validate_smiles)
            counts = df[f'{col}_Status'].value_counts()
            for status, count in counts.items():
                smiles_rows.append({
                    'Column': col,
                    'Status': status,
                    'Count': count,
                    'Percentage': count / len(df) * 100
                })

    smiles_df = pd.DataFrame(smiles_rows)
    smiles_df.to_excel(base_dir / 'SMILES_validation_results.xlsx', index=False)
    plot_smiles_validation(smiles_df, dirs)

    df.to_excel(base_dir / 'processed_carbon_dots_data.xlsx', index=False)
    write_report(base_dir, dirs, df, corr_vars)

    print(f'All outputs saved to: {base_dir} - data_distribution_journal_consistent.py:896')

if __name__ == '__main__':
    main()