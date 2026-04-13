# -*- coding: utf-8 -*-
"""
碳点数据库建模脚本（增强可视化 + 分类整理版）
------------------------------------------------
功能补充：
1. 不同模型 R2 对比
2. 各目标最佳 R2 汇总
3. 最佳模型线性拟合图
4. 残差分析图
5. R2 / MAE / RMSE 柱状图
6. 特征重要性分析图
7. 图像按目录分类保存
"""

import os
import re
import math
import time
import random
import warnings
from typing import Dict, List, Any, Tuple

warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DEFAULT_ROOT_DIR = r"E:\桌面\不惜一切代价"
DEFAULT_DATA_NAME = "data_fixed_with_PCA_smiles_cleaned_corrected_modified.xlsx"

if os.path.exists(DEFAULT_ROOT_DIR) or os.name == 'nt':
    ROOT_DIR = DEFAULT_ROOT_DIR
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DATA_PATH = os.path.join(ROOT_DIR, DEFAULT_DATA_NAME)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results_locked')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures_locked')
MODELS_DIR = os.path.join(ROOT_DIR, 'models_locked')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports_locked')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions_locked')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
IMPORTANCE_DIR = os.path.join(RESULTS_DIR, 'importance_tables')

SUMMARY_EXCEL_PATH = os.path.join(ROOT_DIR, 'model_summary_locked_visual_plus.xlsx')
RUNTIME_SUMMARY_PATH = os.path.join(REPORTS_DIR, 'runtime_summary_visual_plus.txt')
RUN_LOG_PATH = os.path.join(REPORTS_DIR, 'runtime_log_visual_plus.txt')
NEW_TEMPLATE_PATH = os.path.join(ROOT_DIR, 'new_OPD_prediction_template_locked.xlsx')
NEW_PREDICTIONS_PATH = os.path.join(ROOT_DIR, 'new_OPD_predictions_locked.xlsx')

TARGETS = ['quantum_yield', 'ex', 'em']
TARGET_DISPLAY = {'quantum_yield': 'Quantum Yield', 'ex': 'Ex', 'em': 'Em'}
PASS_THRESHOLDS = {'quantum_yield': 0.85, 'ex': 0.85, 'em': 0.85}
SEED_SWEEP = [0]

FIG_DIRS = {
    'overview': os.path.join(FIGURES_DIR, '00_overview'),
    'r2_compare': os.path.join(FIGURES_DIR, '01_model_r2_compare'),
    'best_r2': os.path.join(FIGURES_DIR, '02_best_r2'),
    'linear_fit': os.path.join(FIGURES_DIR, '03_linear_fit'),
    'residual': os.path.join(FIGURES_DIR, '04_residual_analysis'),
    'metrics_bar': os.path.join(FIGURES_DIR, '05_metrics_bar'),
    'importance': os.path.join(FIGURES_DIR, '06_importance_analysis'),
}

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
}

def apply_journal_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'font.weight': 'bold',

        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 1.4,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',

        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,

        'legend.fontsize': 8.5,
        'legend.frameon': False,

        'figure.dpi': 200,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'axes.unicode_minus': False,
    })

def style_axes(ax, add_grid: bool = False, grid_axis: str = 'y'):
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.4)
        ax.spines[spine].set_color('black')

    # 删除所有背景虚线/网格
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(False)

    # 只保留下、左刻度，不在上、右显示刻度
    ax.tick_params(
        axis='both',
        which='both',
        top=False,
        right=False,
        width=1.2,
        length=4,
        labelsize=9
    )

    # 加粗刻度文字
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')

    # 加粗标题和坐标轴标题
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.title.set_fontweight('bold')

def style_colorbar(cbar):
    if cbar is None:
        return
    try:
        cbar.ax.yaxis.label.set_fontweight('bold')
    except Exception:
        pass
    try:
        for tick_label in cbar.ax.get_yticklabels():
            tick_label.set_fontweight('bold')
    except Exception:
        pass

def bold_legend(legend):
    if legend is None:
        return
    try:
        if legend.get_title() is not None:
            legend.get_title().set_fontweight('bold')
    except Exception:
        pass
    for text in legend.get_texts():
        text.set_fontweight('bold')

def add_panel_label(ax, label: str):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')

def annotate_bars(ax, bars, fmt: str = '{:.3f}', offset_ratio: float = 0.02):
    heights = [float(getattr(bar, 'get_height', lambda: 0)()) for bar in bars]
    ymax = max(heights) if heights else 1.0
    offset = max(ymax * offset_ratio, 0.003)
    for bar in bars:
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )

def target_label(name: str) -> str:
    return {'quantum_yield': 'Quantum Yield', 'ex': 'Excitation Wavelength', 'em': 'Emission Wavelength'}.get(name, name)

apply_journal_style()

DESCRIPTOR_NAMES = [
    'MolWt', 'MolLogP', 'TPSA', 'LabuteASA', 'FractionCSP3',
    'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
    'NumRotatableBonds', 'NumAromaticRings', 'RingCount', 'MolMR', 'BertzCT',
    'Kappa1', 'Kappa2', 'Kappa3'
]

class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('=== runtime log ===\n')

    def write(self, msg: str):
        text = str(msg)
        print(text)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

LOGGER = Logger(RUN_LOG_PATH)

def ensure_dirs():
    for p in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, PREDICTIONS_DIR, TABLES_DIR, IMPORTANCE_DIR]:
        os.makedirs(p, exist_ok=True)
    for p in FIG_DIRS.values():
        os.makedirs(p, exist_ok=True)

def savefig(path: str, fig=None, dpi: int = 600):
    fig = fig or plt.gcf()
    fig.tight_layout()
    base, _ = os.path.splitext(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(base + '.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(base + '.pdf', bbox_inches='tight', facecolor='white')
    try:
        fig.savefig(base + '.tiff', dpi=dpi, bbox_inches='tight', facecolor='white')
    except Exception:
        pass
    plt.close(fig)

def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r'(-?\d+(?:\.\d+)?)')[0], errors='coerce')

def normalize_text(text: str) -> str:
    s = str(text).strip().lower()
    s = s.replace('\n', ' ')
    s = s.replace('（', '(').replace('）', ')')
    s = s.replace('℃', 'c')
    s = re.sub(r'[%\s_/\\\-]+', '', s)
    s = s.replace('(', '').replace(')', '').replace('.', '')
    s = s.replace('量子产率', 'quantumyield').replace('时间', 'time').replace('温度', 'temperature').replace('体积', 'volume')
    return s

def infer_standard_columns(df: pd.DataFrame) -> Dict[str, Any]:
    col_map = {c: normalize_text(c) for c in df.columns}
    smiles_cols, amount_cols = [], []
    for raw, norm in col_map.items():
        if 'smiles' in norm:
            smiles_cols.append(raw)
        if ('material' in norm or 'precursor' in norm or '前驱体' in raw) and ('m' in norm or 'amount' in norm or '浓度' in raw) and 'smiles' not in norm:
            amount_cols.append(raw)

    def get_idx(name: str) -> int:
        nums = re.findall(r'\d+', str(name))
        return int(nums[0]) if nums else 999

    smiles_cols = sorted(smiles_cols, key=get_idx)
    amount_cols = sorted(amount_cols, key=get_idx)

    def pick(*keys):
        for raw, norm in col_map.items():
            if all(k in norm for k in keys):
                return raw
        return None

    standard = {
        'smiles_cols': smiles_cols,
        'amount_cols': amount_cols,
        'sample_name': pick('name'),
        'solvent': pick('solvent'),
        'solvent_pc1': pick('solvent', 'pc1'),
        'solvent_pc2': pick('solvent', 'pc2'),
        'volume': pick('volume'),
        'temperature': pick('temperature'),
        'time': pick('time'),
        'quantum_yield': None,
        'ex': None,
        'em': None,
        'particle_size': None,
        'stoke': None,
    }
    for raw, norm in col_map.items():
        if 'quantumyield' in norm and not norm.endswith('original'):
            standard['quantum_yield'] = raw
        elif (norm.startswith('ex') or 'exnm' in norm) and not norm.endswith('original'):
            standard['ex'] = raw
        elif (norm.startswith('em') or 'emnm' in norm) and not norm.endswith('original'):
            standard['em'] = raw
        elif 'particlesize' in norm and not norm.endswith('original'):
            standard['particle_size'] = raw
        elif 'stoke' in norm and not norm.endswith('original'):
            standard['stoke'] = raw
    return standard

def load_and_clean_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'未找到文件: {file_path}')
    raw = pd.read_excel(file_path, sheet_name=0)
    std = infer_standard_columns(raw)

    numeric_candidates = [
        std.get('solvent_pc1'), std.get('solvent_pc2'), std.get('volume'), std.get('temperature'), std.get('time'),
        std.get('quantum_yield'), std.get('ex'), std.get('em'), std.get('particle_size'), std.get('stoke')
    ] + std.get('amount_cols', [])
    numeric_candidates = [c for c in numeric_candidates if c and c in raw.columns]
    for c in numeric_candidates:
        raw[c] = safe_to_numeric(raw[c])

    df = pd.DataFrame(index=raw.index)
    df['sample_name'] = raw[std['sample_name']] if std['sample_name'] else [f'sample_{i}' for i in range(len(raw))]
    df['solvent'] = raw[std['solvent']] if std['solvent'] else np.nan
    df['solvent_pc1'] = raw[std['solvent_pc1']] if std['solvent_pc1'] else np.nan
    df['solvent_pc2'] = raw[std['solvent_pc2']] if std['solvent_pc2'] else np.nan
    df['volume'] = raw[std['volume']] if std['volume'] else np.nan
    df['temperature'] = raw[std['temperature']] if std['temperature'] else np.nan
    df['time'] = raw[std['time']] if std['time'] else np.nan
    df['quantum_yield'] = raw[std['quantum_yield']] if std['quantum_yield'] else np.nan
    df['ex'] = raw[std['ex']] if std['ex'] else np.nan
    df['em'] = raw[std['em']] if std['em'] else np.nan
    df['particle_size'] = raw[std['particle_size']] if std['particle_size'] else np.nan
    df['stoke'] = raw[std['stoke']] if std['stoke'] else np.nan

    max_precursor = max(len(std['smiles_cols']), len(std['amount_cols']), 1)
    for i in range(max_precursor):
        s_col = std['smiles_cols'][i] if i < len(std['smiles_cols']) else None
        a_col = std['amount_cols'][i] if i < len(std['amount_cols']) else None
        df[f'precursor_{i+1}_smiles'] = raw[s_col] if s_col else np.nan
        df[f'precursor_{i+1}_amount_m'] = raw[a_col] if a_col else np.nan

    amount_cols = [f'precursor_{i+1}_amount_m' for i in range(max_precursor)]
    smile_cols = [f'precursor_{i+1}_smiles' for i in range(max_precursor)]
    df['precursor_count'] = df[smile_cols].notna().sum(axis=1)
    df['amount_sum'] = df[amount_cols].sum(axis=1, skipna=True)
    df['amount_mean'] = df[amount_cols].replace(0, np.nan).mean(axis=1, skipna=True).fillna(0)
    df['amount_max'] = df[amount_cols].max(axis=1, skipna=True)
    df['amount_min'] = df[amount_cols].replace(0, np.nan).min(axis=1, skipna=True).fillna(0)
    df['temp_time_interaction'] = df['temperature'] * df['time']
    df['temp_volume_interaction'] = df['temperature'] * df['volume']
    df['time_volume_interaction'] = df['time'] * df['volume']

    meta = {'raw_shape': raw.shape, 'max_precursor': max_precursor, 'raw_columns': list(raw.columns)}
    return df, std, meta

def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    s = str(smiles).strip()
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def smiles_to_descriptors(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=float)
    vals = []
    for name in DESCRIPTOR_NAMES:
        try:
            v = getattr(Descriptors, name)(mol)
            vals.append(float(v) if np.isfinite(v) else 0.0)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=float)

def smiles_to_morgan(smiles: str, n_bits: int = 128) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=float)
    gen = GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)

def combine_precursor_features(row: pd.Series, feature_type: str, max_precursor: int) -> np.ndarray:
    smis, amounts = [], []
    for i in range(max_precursor):
        s = row.get(f'precursor_{i+1}_smiles', np.nan)
        a = row.get(f'precursor_{i+1}_amount_m', np.nan)
        if pd.notna(s) and str(s).strip():
            smis.append(str(s).strip())
            amounts.append(0.0 if pd.isna(a) else float(a))
    if feature_type == 'descriptors':
        fn, dim = smiles_to_descriptors, len(DESCRIPTOR_NAMES)
    else:
        fn, dim = smiles_to_morgan, 128
    if not smis:
        return np.zeros(dim, dtype=float)
    w = np.array(amounts, dtype=float)
    w = np.ones(len(smis), dtype=float) / len(smis) if w.sum() <= 0 else w / w.sum()
    mat = np.array([fn(s) for s in smis], dtype=float)
    return np.average(mat, axis=0, weights=w)

def build_smiles_feature_blocks(df: pd.DataFrame, max_precursor: int) -> Dict[str, pd.DataFrame]:
    desc_list, morgan_list = [], []
    for _, row in df.iterrows():
        desc_list.append(combine_precursor_features(row, 'descriptors', max_precursor))
        morgan_list.append(combine_precursor_features(row, 'morgan', max_precursor))
    desc_df = pd.DataFrame(desc_list, columns=[f'mol_desc_{n}' for n in DESCRIPTOR_NAMES])
    morgan_df = pd.DataFrame(morgan_list, columns=[f'mol_morgan_{i}' for i in range(128)])
    return {'descriptors': desc_df, 'morgan': morgan_df, 'morgan_desc': pd.concat([morgan_df, desc_df], axis=1)}

def prepare_xy(df: pd.DataFrame, feature_block: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    base_cols = [
        'solvent_pc1', 'solvent_pc2', 'volume', 'temperature', 'time',
        'precursor_count', 'amount_sum', 'amount_mean', 'amount_max', 'amount_min',
        'temp_time_interaction', 'temp_volume_interaction', 'time_volume_interaction'
    ]
    aux_cols = [c for c in ['particle_size', 'stoke'] if c in df.columns]
    use_cols = [c for c in base_cols + aux_cols if c in df.columns]
    X = pd.concat([df[use_cols].reset_index(drop=True), feature_block.reset_index(drop=True)], axis=1)
    X.columns = X.columns.astype(str)
    y = df[target_col].astype(float)
    return X, y

def build_models(seed: int) -> Dict[str, Any]:
    return {
        'ExtraTrees': ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1, max_features='sqrt', min_samples_leaf=1),
        'RandomForest': RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1, max_features='sqrt', min_samples_leaf=1),
        'SVR': SVR(C=18, gamma='scale', epsilon=0.03, kernel='rbf'),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance', p=2),
    }

def build_pipeline(model, k: int):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('variance', VarianceThreshold(0.0)),
        ('selector', SelectKBest(score_func=f_regression, k=k)),
        ('scaler', RobustScaler()),
        ('model', model),
    ])

def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def get_selected_feature_names(pipe: Pipeline, X_columns: List[str]) -> List[str]:
    names = np.array(X_columns, dtype=object)
    if hasattr(pipe.named_steps['variance'], 'get_support'):
        names = names[pipe.named_steps['variance'].get_support()]
    if hasattr(pipe.named_steps['selector'], 'get_support'):
        names = names[pipe.named_steps['selector'].get_support()]
    return list(names)

def train_target(df: pd.DataFrame, feature_blocks: Dict[str, pd.DataFrame], target_name: str) -> Dict[str, Any]:
    best_result = None
    records = []
    feature_block_name = 'descriptors'
    feature_block = feature_blocks[feature_block_name]
    X, y = prepare_xy(df, feature_block, target_name)
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    if len(y) < 20:
        return {'best': None, 'records': pd.DataFrame()}

    k = max(8, min(36, X.shape[1] - 1))
    for seed in SEED_SWEEP:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for model_name, model in build_models(seed).items():
            pipe = build_pipeline(model, k=k)
            pipe.fit(X_train, y_train)
            pred_train = pipe.predict(X_train)
            pred_test = pipe.predict(X_test)
            m_train = evaluate_predictions(y_train, pred_train)
            m_test = evaluate_predictions(y_test, pred_test)
            result = {
                'target': target_name,
                'model_name': model_name,
                'feature_block': feature_block_name,
                'seed': seed,
                'train_r2': m_train['r2'],
                'test_r2': m_test['r2'],
                'mae': m_test['mae'],
                'rmse': m_test['rmse'],
                'best_estimator': pipe,
                'X_columns': list(X.columns),
                'selected_feature_names': get_selected_feature_names(pipe, list(X.columns)),
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': pred_test,
            }
            records.append({
                'target': target_name,
                'model_name': model_name,
                'feature_block': feature_block_name,
                'seed': seed,
                'train_r2': m_train['r2'],
                'test_r2': m_test['r2'],
                'mae': m_test['mae'],
                'rmse': m_test['rmse'],
                'pass_threshold': PASS_THRESHOLDS[target_name],
                'pass_flag': int(m_test['r2'] >= PASS_THRESHOLDS[target_name])
            })
            LOGGER.write(f'[{target_name}] {model_name} | test_r2={m_test["r2"]:.4f}')
            if best_result is None or result['test_r2'] > best_result['test_r2']:
                best_result = result
    return {'best': best_result, 'records': pd.DataFrame(records)}

def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ['solvent_pc1', 'solvent_pc2', 'volume', 'temperature', 'time', 'amount_sum', 'amount_mean', 'particle_size', 'stoke', 'quantum_yield', 'ex', 'em']
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    im = ax.imshow(corr.values, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson correlation coefficient', fontweight='bold')
    style_colorbar(cbar)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(cols, fontweight='bold')
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            ax.text(
                j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, fontweight='bold',
                color='white' if abs(val) > 0.55 else 'black'
            )
    ax.set_title('Correlation matrix of numerical variables', fontweight='bold')
    style_axes(ax, add_grid=False)
    savefig(os.path.join(FIG_DIRS['overview'], '01_correlation_heatmap.png'), fig=fig)

def plot_target_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    for idx, (ax, target) in enumerate(zip(axes, TARGETS)):
        vals = df[target].dropna().values
        if len(vals) == 0:
            ax.axis('off')
            continue
        ax.hist(vals, bins=20, color=JOURNAL_COLORS['primary'], edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.axvline(np.mean(vals), color=JOURNAL_COLORS['accent'], linestyle='--', linewidth=1.2, label=f'Mean = {np.mean(vals):.2f}')
        ax.axvline(np.median(vals), color=JOURNAL_COLORS['green'], linestyle=':', linewidth=1.2, label=f'Median = {np.median(vals):.2f}')
        ax.set_title(target_label(target), fontweight='bold')
        ax.set_xlabel('Observed value', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        style_axes(ax, add_grid=False)
        leg = ax.legend(loc='upper right')
        bold_legend(leg)
        add_panel_label(ax, chr(65 + idx))
    savefig(os.path.join(FIG_DIRS['overview'], '02_target_distributions.png'), fig=fig)

def plot_numeric_feature_distributions(df: pd.DataFrame):
    cols = ['temperature', 'time', 'volume', 'solvent_pc1', 'solvent_pc2', 'amount_sum', 'precursor_count', 'particle_size', 'stoke']
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    ncols, nrows = 3, max(1, math.ceil(len(cols) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        ax = axes[i]
        vals = df[col].dropna().values
        ax.hist(vals, bins=18, color=JOURNAL_COLORS['secondary'], edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.axvline(np.nanmean(vals), color=JOURNAL_COLORS['accent'], linestyle='--', linewidth=1.1)
        ax.set_title(col.replace('_', ' '), fontweight='bold')
        ax.set_xlabel('Value', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        style_axes(ax, add_grid=False)
    for j in range(len(cols), len(axes)):
        axes[j].axis('off')
    savefig(os.path.join(FIG_DIRS['overview'], '03_numeric_feature_distributions.png'), fig=fig)

def plot_target_pair_scatter(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    pairs = [('ex', 'em'), ('quantum_yield', 'ex'), ('quantum_yield', 'em')]
    for idx, (ax, (xcol, ycol)) in enumerate(zip(axes, pairs)):
        sub = df[[xcol, ycol]].dropna()
        if sub.empty:
            ax.axis('off')
            continue
        ax.scatter(
            sub[xcol], sub[ycol], s=28, alpha=0.8, facecolor=JOURNAL_COLORS['light_blue'],
            edgecolor=JOURNAL_COLORS['primary'], linewidth=0.7
        )
        if len(sub) > 1:
            coef = np.polyfit(sub[xcol], sub[ycol], 1)
            fit_x = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
            fit_y = np.polyval(coef, fit_x)
            corr = np.corrcoef(sub[xcol], sub[ycol])[0, 1]
            ax.plot(fit_x, fit_y, color=JOURNAL_COLORS['accent'], linewidth=1.5)
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, va='top', fontweight='bold')
        ax.set_xlabel(target_label(xcol), fontweight='bold')
        ax.set_ylabel(target_label(ycol), fontweight='bold')
        ax.set_title(f'{target_label(xcol)} vs. {target_label(ycol)}', fontweight='bold')
        style_axes(ax, add_grid=False)
        add_panel_label(ax, chr(65 + idx))
    savefig(os.path.join(FIG_DIRS['overview'], '04_target_pair_scatter.png'), fig=fig)

def plot_pca_map(feature_blocks: Dict[str, pd.DataFrame], df: pd.DataFrame):
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(feature_blocks['descriptors'].fillna(0))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    for idx, (ax, target) in enumerate(zip(axes, TARGETS)):
        valid = ~df[target].isna()
        sc = ax.scatter(
            coords[valid, 0], coords[valid, 1], c=df.loc[valid, target], s=26,
            cmap='viridis', alpha=0.85, edgecolor='black', linewidth=0.3
        )
        ax.set_title(f'PCA map colored by {target_label(target)}', fontweight='bold')
        ax.set_xlabel('PC1', fontweight='bold')
        ax.set_ylabel('PC2', fontweight='bold')
        cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.03)
        cbar.set_label(target_label(target), fontweight='bold')
        style_colorbar(cbar)
        style_axes(ax, add_grid=False)
        add_panel_label(ax, chr(65 + idx))
    savefig(os.path.join(FIG_DIRS['overview'], '05_molecular_pca_map.png'), fig=fig)

def plot_model_score_heatmap(all_records: pd.DataFrame):
    if all_records.empty:
        return
    pivot = all_records.groupby(['target', 'model_name'])['test_r2'].max().reindex(TARGETS, level=0).unstack()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    vals = pivot.values.astype(float)
    im = ax.imshow(vals, cmap='YlGnBu', aspect='auto', vmin=max(0.0, np.nanmin(vals) - 0.05), vmax=max(1.0, np.nanmax(vals)))
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('Test $R^2$', fontweight='bold')
    style_colorbar(cbar)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha='right', fontweight='bold')
    ax.set_yticklabels([target_label(x) for x in pivot.index], fontweight='bold')
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(
                    j, i, f'{vals[i, j]:.3f}', ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white' if vals[i, j] > 0.82 else 'black'
                )
    ax.set_title('Heatmap of best model performance across targets', fontweight='bold')
    style_axes(ax, add_grid=False)
    savefig(os.path.join(FIG_DIRS['r2_compare'], '01_model_score_heatmap.png'), fig=fig)

def plot_model_r2_comparison(all_records: pd.DataFrame):
    if all_records.empty:
        return
    pivot = all_records.groupby(['model_name', 'target'])['test_r2'].max().unstack().reindex(sorted(all_records['model_name'].unique()))
    x = np.arange(len(pivot.index))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    colors = [JOURNAL_COLORS['primary'], JOURNAL_COLORS['green'], JOURNAL_COLORS['gold']]
    for idx, target in enumerate(TARGETS):
        vals = pivot[target].values if target in pivot.columns else np.zeros(len(pivot.index))
        ax.bar(
            x + (idx - 1) * width, vals, width=width, color=colors[idx], edgecolor='black',
            linewidth=0.7, label=target_label(target), alpha=0.9
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha='right', fontweight='bold')
    ax.set_ylabel('Test $R^2$', fontweight='bold')
    ax.set_ylim(0, max(1.0, float(np.nanmax(pivot.values)) * 1.12))
    ax.set_title('Comparison of model $R^2$ across all targets', fontweight='bold')
    style_axes(ax, add_grid=False)
    leg = ax.legend(loc='upper left', ncol=3)
    bold_legend(leg)
    savefig(os.path.join(FIG_DIRS['r2_compare'], '02_all_targets_model_r2_compare.png'), fig=fig)

    palette = [JOURNAL_COLORS['primary'], JOURNAL_COLORS['green'], JOURNAL_COLORS['gold'], JOURNAL_COLORS['purple']]
    for target in TARGETS:
        sub = all_records[all_records['target'] == target].sort_values('test_r2', ascending=False).drop_duplicates('model_name')
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        bars = ax.bar(sub['model_name'], sub['test_r2'], color=palette[:len(sub)], edgecolor='black', linewidth=0.8)
        ax.axhline(
            PASS_THRESHOLDS[target], linestyle='--', linewidth=1.3, color=JOURNAL_COLORS['accent'],
            label=f'Threshold = {PASS_THRESHOLDS[target]:.2f}'
        )
        annotate_bars(ax, bars, fmt='{:.4f}')
        ax.set_ylabel('Test $R^2$', fontweight='bold')
        ax.set_ylim(0, max(1.0, float(sub['test_r2'].max()) * 1.14))
        ax.set_title(f'{target_label(target)}: model-wise $R^2$ comparison', fontweight='bold')
        style_axes(ax, add_grid=False)
        leg = ax.legend(loc='upper right')
        bold_legend(leg)
        savefig(os.path.join(FIG_DIRS['r2_compare'], f'03_{target}_model_r2_compare.png'), fig=fig)

def plot_best_r2_summary(best_df: pd.DataFrame):
    if best_df.empty:
        return
    ordered = best_df.set_index('target').reindex(TARGETS).reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(
        [target_label(t) for t in ordered['target']], ordered['test_r2'],
        color=[JOURNAL_COLORS['primary'], JOURNAL_COLORS['green'], JOURNAL_COLORS['gold']],
        edgecolor='black', linewidth=0.8
    )
    for i, row in ordered.iterrows():
        ax.text(
            i, row['test_r2'] + 0.015, f"{row['best_model_name']}\n$R^2$ = {row['test_r2']:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )
    for row in ordered.itertuples():
        ax.axhline(row.pass_threshold, linestyle='--', linewidth=0.9, color=JOURNAL_COLORS['accent'], alpha=0.45)
    ax.set_ylabel('Best test $R^2$', fontweight='bold')
    ax.set_ylim(0, max(1.0, float(ordered['test_r2'].max()) * 1.16))
    ax.set_title('Best predictive performance for each target', fontweight='bold')
    style_axes(ax, add_grid=False)
    savefig(os.path.join(FIG_DIRS['best_r2'], '01_best_r2_summary.png'), fig=fig)

    ranked = best_df.sort_values('test_r2', ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(ranked['target'].map(target_label), ranked['test_r2'], color=JOURNAL_COLORS['secondary'], edgecolor='black', linewidth=0.8)
    annotate_bars(ax, bars, fmt='{:.4f}')
    ax.set_ylabel('Test $R^2$', fontweight='bold')
    ax.set_ylim(0, max(1.0, float(ranked['test_r2'].max()) * 1.14))
    ax.set_title('Ranking of best target-specific models', fontweight='bold')
    style_axes(ax, add_grid=False)
    savefig(os.path.join(FIG_DIRS['best_r2'], '02_best_r2_ranking.png'), fig=fig)

def plot_linear_fit(target_name: str, best_result: Dict[str, Any]):
    if best_result is None:
        return
    y_true = np.asarray(best_result['y_test'], dtype=float)
    y_pred = np.asarray(best_result['y_pred'], dtype=float)
    if len(y_true) < 2:
        return
    coef = np.polyfit(y_true, y_pred, 1)
    fit_x = np.linspace(float(np.min(y_true)), float(np.max(y_true)), 200)
    fit_y = np.polyval(coef, fit_x)
    lo, hi = min(float(np.min(y_true)), float(np.min(y_pred))), max(float(np.max(y_true)), float(np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(
        y_true, y_pred, s=36, alpha=0.85, facecolor=JOURNAL_COLORS['light_blue'],
        edgecolor=JOURNAL_COLORS['primary'], linewidth=0.8
    )
    ax.plot([lo, hi], [lo, hi], '--', linewidth=1.3, color=JOURNAL_COLORS['secondary'], label='Ideal fit ($y=x$)')
    ax.plot(fit_x, fit_y, linewidth=1.8, color=JOURNAL_COLORS['accent'], label='Linear regression')
    eq = f'$y$ = {coef[0]:.3f}$x$ + {coef[1]:.3f}'
    txt = '\n'.join([
        f"Model: {best_result['model_name']}",
        f"Test $R^2$ = {best_result['test_r2']:.4f}",
        f"MAE = {best_result['mae']:.4f}",
        f"RMSE = {best_result['rmse']:.4f}",
        eq,
    ])
    ax.text(
        0.04, 0.96, txt, transform=ax.transAxes, va='top', fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.92)
    )
    ax.set_xlabel('Observed value', fontweight='bold')
    ax.set_ylabel('Predicted value', fontweight='bold')
    ax.set_title(f'{target_label(target_name)}: observed vs. predicted', fontweight='bold')
    style_axes(ax, add_grid=False)
    leg = ax.legend(loc='lower right')
    bold_legend(leg)
    savefig(os.path.join(FIG_DIRS['linear_fit'], f'01_linear_fit_{target_name}.png'), fig=fig)

def plot_residual_analysis(target_name: str, best_result: Dict[str, Any]):
    if best_result is None:
        return
    y_true = np.asarray(best_result['y_test'], dtype=float)
    y_pred = np.asarray(best_result['y_pred'], dtype=float)
    residuals = y_true - y_pred
    std_res = residuals / (np.std(residuals) + 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.scatter(
        y_pred, residuals, s=34, alpha=0.82, facecolor=JOURNAL_COLORS['light_blue'],
        edgecolor=JOURNAL_COLORS['primary'], linewidth=0.7
    )
    ax1.axhline(0, linestyle='--', linewidth=1.2, color=JOURNAL_COLORS['accent'])
    ax1.set_xlabel('Predicted value', fontweight='bold')
    ax1.set_ylabel('Residual', fontweight='bold')
    ax1.set_title('Residuals vs. predicted', fontweight='bold')
    style_axes(ax1, add_grid=False)
    add_panel_label(ax1, 'A')

    ax2.scatter(
        y_true, residuals, s=34, alpha=0.82, facecolor=JOURNAL_COLORS['light_red'],
        edgecolor=JOURNAL_COLORS['accent'], linewidth=0.7
    )
    ax2.axhline(0, linestyle='--', linewidth=1.2, color=JOURNAL_COLORS['secondary'])
    ax2.set_xlabel('Observed value', fontweight='bold')
    ax2.set_ylabel('Residual', fontweight='bold')
    ax2.set_title('Residuals vs. observed', fontweight='bold')
    style_axes(ax2, add_grid=False)
    add_panel_label(ax2, 'B')

    ax3.hist(residuals, bins=16, color=JOURNAL_COLORS['secondary'], edgecolor='black', linewidth=0.8, alpha=0.85)
    ax3.axvline(np.mean(residuals), linestyle='--', linewidth=1.2, color=JOURNAL_COLORS['accent'], label=f'Mean = {np.mean(residuals):.3f}')
    ax3.axvline(0, linestyle=':', linewidth=1.0, color='black')
    ax3.set_xlabel('Residual', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Residual distribution', fontweight='bold')
    style_axes(ax3, add_grid=False)
    leg = ax3.legend(loc='upper right')
    bold_legend(leg)
    add_panel_label(ax3, 'C')

    order = np.argsort(y_pred)
    ax4.plot(
        np.arange(len(std_res)), std_res[order], marker='o', markersize=3.8, linewidth=1.1,
        color=JOURNAL_COLORS['purple'], markerfacecolor='white'
    )
    ax4.axhline(0, linestyle='--', linewidth=1.2, color=JOURNAL_COLORS['accent'])
    ax4.axhline(2, linestyle=':', linewidth=1.0, color=JOURNAL_COLORS['secondary'])
    ax4.axhline(-2, linestyle=':', linewidth=1.0, color=JOURNAL_COLORS['secondary'])
    ax4.set_xlabel('Samples sorted by predicted value', fontweight='bold')
    ax4.set_ylabel('Standardized residual', fontweight='bold')
    ax4.set_title('Standardized residual trend', fontweight='bold')
    style_axes(ax4, add_grid=False)
    add_panel_label(ax4, 'D')

    fig.suptitle(f'{target_label(target_name)}: residual diagnostics', fontsize=11, fontweight='bold', y=1.01)
    savefig(os.path.join(FIG_DIRS['residual'], f'01_residual_analysis_{target_name}.png'), fig=fig)

def plot_metric_bars_for_target(target_name: str, records_df: pd.DataFrame):
    sub = records_df[records_df['target'] == target_name].copy()
    if sub.empty:
        return
    sub = sub.sort_values('test_r2', ascending=False).drop_duplicates('model_name')
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))

    metric_specs = [
        ('test_r2', 'Test $R^2$', JOURNAL_COLORS['primary']),
        ('mae', 'MAE', JOURNAL_COLORS['green']),
        ('rmse', 'RMSE', JOURNAL_COLORS['gold']),
    ]
    for idx, (ax, (metric, title, color)) in enumerate(zip(axes, metric_specs)):
        bars = ax.bar(sub['model_name'], sub[metric], color=color, edgecolor='black', linewidth=0.8, alpha=0.9)
        if metric == 'test_r2':
            ax.axhline(PASS_THRESHOLDS[target_name], linestyle='--', linewidth=1.2, color=JOURNAL_COLORS['accent'])
            ax.set_ylim(0, max(1.0, float(sub[metric].max()) * 1.12))
        else:
            ymax = float(sub[metric].max())
            ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
        annotate_bars(ax, bars, fmt='{:.3f}')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
        style_axes(ax, add_grid=False)
        add_panel_label(ax, chr(65 + idx))
    fig.suptitle(f'{target_label(target_name)}: model comparison by evaluation metric', fontsize=11, fontweight='bold', y=1.02)
    savefig(os.path.join(FIG_DIRS['metrics_bar'], f'01_metrics_bar_{target_name}.png'), fig=fig)

def plot_best_metric_summary(best_df: pd.DataFrame):
    if best_df.empty:
        return
    ordered = best_df.set_index('target').reindex(TARGETS).reset_index()
    x = np.arange(len(ordered))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x - width, ordered['test_r2'], width=width, color=JOURNAL_COLORS['primary'], edgecolor='black', linewidth=0.8, label='Test $R^2$')
    ax.bar(x, ordered['mae'], width=width, color=JOURNAL_COLORS['green'], edgecolor='black', linewidth=0.8, label='MAE')
    ax.bar(x + width, ordered['rmse'], width=width, color=JOURNAL_COLORS['gold'], edgecolor='black', linewidth=0.8, label='RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels([target_label(t) for t in ordered['target']], fontweight='bold')
    ax.set_ylabel('Metric value', fontweight='bold')
    ax.set_title('Performance summary of the best model for each target', fontweight='bold')
    style_axes(ax, add_grid=False)
    leg = ax.legend(loc='upper right', ncol=3)
    bold_legend(leg)
    savefig(os.path.join(FIG_DIRS['metrics_bar'], '02_best_metric_summary.png'), fig=fig)

def compute_importance_df(best_result: Dict[str, Any]) -> pd.DataFrame:
    pipe = best_result['best_estimator']
    names = best_result.get('selected_feature_names') or get_selected_feature_names(pipe, best_result['X_columns'])
    if not names:
        return pd.DataFrame(columns=['feature', 'importance', 'importance_type'])
    model = pipe.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        vals = np.asarray(model.feature_importances_, dtype=float)
        out = pd.DataFrame({'feature': names[:len(vals)], 'importance': vals[:len(names)], 'importance_type': 'native_tree_importance'})
        return out.sort_values('importance', ascending=False).head(20).reset_index(drop=True)

    try:
        X_temp = best_result['X_test'].copy()
        X_temp = pipe.named_steps['imputer'].transform(X_temp)
        X_temp = pipe.named_steps['variance'].transform(X_temp)
        X_temp = pipe.named_steps['selector'].transform(X_temp)
        X_temp = pipe.named_steps['scaler'].transform(X_temp)
        X_temp = np.asarray(X_temp, dtype=float)
        y_temp = np.asarray(best_result['y_test'], dtype=float)
        scores = []
        for i in range(X_temp.shape[1]):
            col = X_temp[:, i]
            if np.std(col) == 0 or np.std(y_temp) == 0:
                scores.append(0.0)
            else:
                scores.append(abs(float(np.corrcoef(col, y_temp)[0, 1])))
        out = pd.DataFrame({'feature': names[:len(scores)], 'importance': scores[:len(names)], 'importance_type': 'abs_correlation_proxy'})
        return out.sort_values('importance', ascending=False).head(20).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['feature', 'importance', 'importance_type'])

def plot_feature_importance(target_name: str, best_result: Dict[str, Any]):
    imp_df = compute_importance_df(best_result)
    if imp_df.empty:
        return
    imp_df.to_excel(os.path.join(IMPORTANCE_DIR, f'importance_{target_name}.xlsx'), index=False)
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ordered = imp_df.iloc[::-1]
    ax.barh(ordered['feature'], ordered['importance'], color=JOURNAL_COLORS['primary'], edgecolor='black', linewidth=0.7, alpha=0.9)
    ax.set_xlabel('Relative importance', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    imp_type = imp_df['importance_type'].iloc[0].replace('_', ' ')
    ax.set_title(f'{target_label(target_name)}: top 20 feature importance ({imp_type})', fontweight='bold')
    style_axes(ax, add_grid=False)
    savefig(os.path.join(FIG_DIRS['importance'], f'01_importance_{target_name}.png'), fig=fig)

def create_prediction_template():
    template = pd.DataFrame({
        'sample_name': ['new_sample_1', 'new_sample_2'],
        'precursor_1_smiles': ['c1ccc(c(c1)N)N', 'Nc1ccccc1'],
        'precursor_1_amount_m': [0.5, 0.2],
        'precursor_2_smiles': ['', ''],
        'precursor_2_amount_m': ['', ''],
        'precursor_3_smiles': ['', ''],
        'precursor_3_amount_m': ['', ''],
        'solvent_pc1': [0.0, 2.4],
        'solvent_pc2': [0.0, -0.5],
        'volume': [10.0, 20.0],
        'temperature': [200.0, 180.0],
        'time': [8.0, 12.0],
        'particle_size': ['', ''],
        'stoke': ['', ''],
    })
    template.to_excel(NEW_TEMPLATE_PATH, index=False)

def build_new_feature_df(new_df: pd.DataFrame, feature_block_name: str) -> pd.DataFrame:
    temp = new_df.copy()
    for c in ['solvent_pc1', 'solvent_pc2', 'volume', 'temperature', 'time', 'particle_size', 'stoke', 'precursor_1_amount_m', 'precursor_2_amount_m', 'precursor_3_amount_m']:
        if c in temp.columns:
            temp[c] = safe_to_numeric(temp[c])
    temp['precursor_count'] = temp[['precursor_1_smiles', 'precursor_2_smiles', 'precursor_3_smiles']].notna().sum(axis=1)
    temp['amount_sum'] = temp[['precursor_1_amount_m', 'precursor_2_amount_m', 'precursor_3_amount_m']].sum(axis=1, skipna=True)
    temp['amount_mean'] = temp[['precursor_1_amount_m', 'precursor_2_amount_m', 'precursor_3_amount_m']].replace(0, np.nan).mean(axis=1, skipna=True).fillna(0)
    temp['amount_max'] = temp[['precursor_1_amount_m', 'precursor_2_amount_m', 'precursor_3_amount_m']].max(axis=1, skipna=True)
    temp['amount_min'] = temp[['precursor_1_amount_m', 'precursor_2_amount_m', 'precursor_3_amount_m']].replace(0, np.nan).min(axis=1, skipna=True).fillna(0)
    temp['temp_time_interaction'] = temp['temperature'] * temp['time']
    temp['temp_volume_interaction'] = temp['temperature'] * temp['volume']
    temp['time_volume_interaction'] = temp['time'] * temp['volume']
    block_df = build_smiles_feature_blocks(temp, max_precursor=3)[feature_block_name]
    base_cols = ['solvent_pc1', 'solvent_pc2', 'volume', 'temperature', 'time', 'precursor_count', 'amount_sum', 'amount_mean', 'amount_max', 'amount_min', 'temp_time_interaction', 'temp_volume_interaction', 'time_volume_interaction', 'particle_size', 'stoke']
    base_cols = [c for c in base_cols if c in temp.columns]
    X_new = pd.concat([temp[base_cols].reset_index(drop=True), block_df.reset_index(drop=True)], axis=1)
    X_new.columns = X_new.columns.astype(str)
    return X_new

def predict_new_samples(best_map: Dict[str, Dict[str, Any]]):
    if not os.path.exists(NEW_TEMPLATE_PATH):
        create_prediction_template()
        LOGGER.write(f'已创建新样本模板，请先填写后重新运行: {NEW_TEMPLATE_PATH}')
        return

    new_df = pd.read_excel(NEW_TEMPLATE_PATH)
    pred_df = new_df.copy()

    for target_name, info in best_map.items():
        if info is None:
            continue

        bundle_path = os.path.join(MODELS_DIR, f'best_model_{target_name}.joblib')
        if not os.path.exists(bundle_path):
            LOGGER.write(f'模型文件不存在: {bundle_path}')
            continue

        bundle = joblib.load(bundle_path)
        X_new = build_new_feature_df(new_df, bundle['feature_block'])
        pred_df[f'{target_name}_pred'] = bundle['model'].predict(X_new)

    pred_df.to_excel(NEW_PREDICTIONS_PATH, index=False)
    LOGGER.write(f'新样本预测完成，结果已保存: {NEW_PREDICTIONS_PATH}')

def main():
    start = time.time()
    ensure_dirs()
    LOGGER.write(f'ROOT_DIR = {ROOT_DIR}')
    LOGGER.write(f'INPUT_DATA_PATH = {INPUT_DATA_PATH}')

    df, std, meta = load_and_clean_data(INPUT_DATA_PATH)
    feature_blocks = build_smiles_feature_blocks(df, meta['max_precursor'])

    plot_correlation_heatmap(df)
    plot_target_distributions(df)
    plot_numeric_feature_distributions(df)
    plot_target_pair_scatter(df)
    plot_pca_map(feature_blocks, df)

    all_records = []
    best_rows = []
    best_map = {}

    for target in TARGETS:
        pack = train_target(df, feature_blocks, target)
        best_map[target] = pack['best']
        if pack['records'] is not None and not pack['records'].empty:
            all_records.append(pack['records'])
        best = pack['best']
        if best is None:
            continue
        best_rows.append({
            'target': target,
            'best_model_name': best['model_name'],
            'feature_block': best['feature_block'],
            'seed': best['seed'],
            'train_r2': best['train_r2'],
            'test_r2': best['test_r2'],
            'mae': best['mae'],
            'rmse': best['rmse'],
            'pass_threshold': PASS_THRESHOLDS[target],
            'pass_flag': int(best['test_r2'] >= PASS_THRESHOLDS[target]),
        })
        joblib.dump(
            {'model': best['best_estimator'], 'feature_block': best['feature_block'], 'X_columns': best['X_columns']},
            os.path.join(MODELS_DIR, f'best_model_{target}.joblib')
        )

    all_records_df = pd.concat(all_records, axis=0, ignore_index=True) if all_records else pd.DataFrame()
    best_df = pd.DataFrame(best_rows)

    plot_model_score_heatmap(all_records_df)
    plot_model_r2_comparison(all_records_df)
    plot_best_r2_summary(best_df)
    plot_best_metric_summary(best_df)

    for target in TARGETS:
        best = best_map.get(target)
        if best is None:
            continue
        plot_linear_fit(target, best)
        plot_residual_analysis(target, best)
        plot_metric_bars_for_target(target, all_records_df)
        plot_feature_importance(target, best)

    with pd.ExcelWriter(SUMMARY_EXCEL_PATH, engine='openpyxl') as writer:
        if not all_records_df.empty:
            all_records_df.to_excel(writer, index=False, sheet_name='all_records')
        if not best_df.empty:
            best_df.to_excel(writer, index=False, sheet_name='best_results')
        if not all_records_df.empty:
            metrics_by_model = all_records_df.groupby(['target', 'model_name'])[['test_r2', 'mae', 'rmse']].max().reset_index()
            metrics_by_model.to_excel(writer, index=False, sheet_name='metrics_by_model')

    if not all_records_df.empty:
        all_records_df.to_excel(os.path.join(TABLES_DIR, 'all_model_metrics.xlsx'), index=False)
    if not best_df.empty:
        best_df.to_excel(os.path.join(TABLES_DIR, 'best_model_metrics.xlsx'), index=False)

    with open(RUNTIME_SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('Publication-ready visualization run summary\n')
        f.write('=' * 80 + '\n')
        for _, row in best_df.iterrows():
            f.write(f"目标={row['target']} | 最佳模型={row['best_model_name']} | Test R2={row['test_r2']:.4f} | MAE={row['mae']:.4f} | RMSE={row['rmse']:.4f} | 通过={row['pass_flag']}\n")
        f.write('=' * 80 + '\n')
        for key, path in FIG_DIRS.items():
            f.write(f'{key}: {path}\n')
        f.write(f'总耗时: {time.time() - start:.2f} 秒\n')

    predict_new_samples(best_map)
    LOGGER.write('Publication-ready visualization pipeline completed.')

if __name__ == '__main__':
    main()