import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, norm
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置工作目录
import os
os.chdir(r'E:\桌面\数据分布和相关性分析')

# 读取数据
file_path = 'data_fixed_with_PCA.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 清理列名（去除前后空格）
df.columns = df.columns.str.strip()

# 数据预处理
print("数据基本信息： - 数据分布.py:27")
print(f"数据集大小: {df.shape} - 数据分布.py:28")
print(f"\n列名: {df.columns.tolist()} - 数据分布.py:29")

# 数值列转换 - 使用正确的列名
numeric_columns = ['Material 1 (M)', 'Material 2 (M)', 'Material 3 (M)', 
                   'volume (mL)', 'Temperature (℃)', 'Time (h)', 
                   'Quantum yield(%)', 'Ex(nm)', 'Em(nm)', 
                   'particle size(nm)', 'stoke (nm)', 'Abs(nm)',
                   'Solvent_PC1', 'Solvent_PC2']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 创建溶剂类型分类
def classify_solvent(solvent):
    if pd.isna(solvent):
        return 'Unknown'
    solvent_str = str(solvent).lower()
    if 'water' in solvent_str:
        return 'Water'
    elif 'ethanol' in solvent_str:
        return 'Ethanol'
    elif 'dmf' in solvent_str:
        return 'DMF'
    elif 'methanol' in solvent_str:
        return 'Methanol'
    elif 'acetic' in solvent_str:
        return 'Acetic acid'
    else:
        return 'Mixed/Other'

df['Solvent_Type'] = df['Solvent'].apply(classify_solvent)

# 定义函数：保存数据和绘制带拟合曲线的柱状图
def plot_histogram_with_fit(data, column_name, xlabel, ylabel, title, filename, bins=30):
    """绘制带拟合曲线的柱状图并保存数据"""
    # 创建数据DataFrame
    data_df = pd.DataFrame({
        column_name: data,
        'Frequency': np.ones(len(data))
    })
    # 计算直方图数据
    hist_counts, hist_bins = np.histogram(data, bins=bins)
    hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    
    # 保存直方图数据
    hist_data = pd.DataFrame({
        'Bin_Start': hist_bins[:-1],
        'Bin_End': hist_bins[1:],
        'Bin_Center': hist_centers,
        'Frequency': hist_counts,
        'Density': hist_counts / len(data)
    })
    
    # 保存原始数据
    data_df.to_excel(f'{filename}_原始数据.xlsx', index=False)
    hist_data.to_excel(f'{filename}_直方图数据.xlsx', index=False)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, 
                                color='steelblue', density=True, label='实际分布')
    
    # 添加正态分布拟合曲线
    mu, std = norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    y = norm.pdf(x, mu, std)
    ax.plot(x, y, 'r-', linewidth=2, label=f'正态分布拟合\n(μ={mu:.2f}, σ={std:.2f})')
    
    # 添加核密度估计曲线
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    y_kde = kde(x)
    ax.plot(x, y_kde, 'g--', linewidth=2, label='核密度估计')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'{title}\n(n={len(data)})', fontsize=14, fontweight='bold')
    ax.axvline(mu, color='red', linestyle='--', alpha=0.5, label=f'均值: {mu:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return mu, std

# 1. 数据分布特征描述 - 带拟合曲线的柱状图
print("\n - 数据分布.py:119" + "="*80)
print("生成带拟合曲线的柱状图... - 数据分布.py:120")
print("= - 数据分布.py:121"*80)

# 量子产率分布
qy_data = df['Quantum yield(%)'].dropna()
if len(qy_data) > 0:
    mu, std = plot_histogram_with_fit(qy_data, 'Quantum_yield', 
                                       '量子产率 (%)', '密度', 
                                       '量子产率分布特征', 
                                       '数据分布特征_量子产率', bins=30)

# 发射波长分布
em_data = df['Em(nm)'].dropna()
if len(em_data) > 0:
    plot_histogram_with_fit(em_data, 'Em_nm', 
                            '发射波长 (nm)', '密度', 
                            '发射波长分布特征', 
                            '数据分布特征_发射波长', bins=30)

# 斯托克斯位移分布
stoke_data = df['stoke (nm)'].dropna()
if len(stoke_data) > 0:
    plot_histogram_with_fit(stoke_data, 'Stoke_shift_nm', 
                            '斯托克斯位移 (nm)', '密度', 
                            '斯托克斯位移分布特征', 
                            '数据分布特征_斯托克斯位移', bins=30)

# 粒径分布
size_data = df['particle size(nm)'].dropna()
if len(size_data) > 0:
    plot_histogram_with_fit(size_data, 'Particle_size_nm', 
                            '粒径 (nm)', '密度', 
                            '粒径分布特征', 
                            '数据分布特征_粒径', bins=30)

# 温度分布
temp_data = df['Temperature (℃)'].dropna()
if len(temp_data) > 0:
    plot_histogram_with_fit(temp_data, 'Temperature_C', 
                            '温度 (℃)', '密度', 
                            '反应温度分布特征', 
                            '数据分布特征_温度', bins=30)

# 时间分布
time_data = df['Time (h)'].dropna()
if len(time_data) > 0:
    plot_histogram_with_fit(time_data, 'Time_h', 
                            '时间 (h)', '密度', 
                            '反应时间分布特征', 
                            '数据分布特征_时间', bins=30)

# 2. 小提琴图数据保存函数
def save_violin_data_and_plot(data_dict, x_col, y_col, title, filename, figsize=(10, 6)):
    """保存小提琴图数据并绘图"""
    # 合并所有数据
    all_data = []
    for category, values in data_dict.items():
        if len(values) > 0:
            for val in values:
                all_data.append({x_col: category, y_col: val})
    
    plot_df = pd.DataFrame(all_data)
    
    # 保存数据到Excel
    summary_stats = []
    for category, values in data_dict.items():
        if len(values) > 0:
            summary_stats.append({
                'Category': category,
                'Count': len(values),
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Q1': np.percentile(values, 25),
                'Q3': np.percentile(values, 75)
            })
    
    stats_df = pd.DataFrame(summary_stats)
    
    with pd.ExcelWriter(f'{filename}.xlsx') as writer:
        plot_df.to_excel(writer, sheet_name='原始数据', index=False)
        stats_df.to_excel(writer, sheet_name='统计摘要', index=False)
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    categories = list(data_dict.keys())
    data_to_plot = [data_dict[cat] for cat in categories if len(data_dict[cat]) > 0]
    categories_filtered = [cat for cat in categories if len(data_dict[cat]) > 0]
    
    if len(data_to_plot) > 0:
        parts = ax.violinplot(data_to_plot, positions=range(len(categories_filtered)), 
                               showmeans=True, showmedians=True)
        
        # 设置颜色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plt.cm.Set2(i/len(categories_filtered)))
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(categories_filtered)))
        ax.set_xticklabels(categories_filtered, rotation=45, ha='right')
        ax.set_xlabel('分类', fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    return stats_df

# 3. 量子产率的小提琴图
print("\n生成量子产率影响因素小提琴图... - 数据分布.py:235")

# 溶剂类型对QY的影响
solvent_qy_dict = {}
for solvent in df['Solvent_Type'].unique():
    data = df[df['Solvent_Type'] == solvent]['Quantum yield(%)'].dropna().values
    if len(data) > 0:
        solvent_qy_dict[solvent] = data
save_violin_data_and_plot(solvent_qy_dict, 'Solvent_Type', 'Quantum yield(%)', 
                          '不同溶剂类型对量子产率的影响', 
                          '小提琴图_量子产率_溶剂类型')

# 温度分组对QY的影响
df_temp = df[['Temperature (℃)', 'Quantum yield(%)']].dropna()
if len(df_temp) > 0:
    df_temp['Temp_Group'] = pd.cut(df_temp['Temperature (℃)'], bins=[0, 120, 160, 200, 240, 300], 
                                    labels=['<120', '120-160', '160-200', '200-240', '>240'])
    temp_qy_dict = {}
    for temp_group in df_temp['Temp_Group'].unique():
        data = df_temp[df_temp['Temp_Group'] == temp_group]['Quantum yield(%)'].values
        if len(data) > 0:
            temp_qy_dict[str(temp_group)] = data
    save_violin_data_and_plot(temp_qy_dict, 'Temperature_Group', 'Quantum yield(%)', 
                              '不同温度范围对量子产率的影响', 
                              '小提琴图_量子产率_温度范围')

# 时间分组对QY的影响
df_time = df[['Time (h)', 'Quantum yield(%)']].dropna()
if len(df_time) > 0:
    df_time['Time_Group'] = pd.cut(df_time['Time (h)'], bins=[0, 2, 6, 10, 15, 25, 100], 
                                    labels=['<2', '2-6', '6-10', '10-15', '15-25', '>25'])
    time_qy_dict = {}
    for time_group in df_time['Time_Group'].unique():
        data = df_time[df_time['Time_Group'] == time_group]['Quantum yield(%)'].values
        if len(data) > 0:
            time_qy_dict[str(time_group)] = data
    save_violin_data_and_plot(time_qy_dict, 'Time_Group', 'Quantum yield(%)', 
                              '不同反应时间对量子产率的影响', 
                              '小提琴图_量子产率_反应时间')

# 材料数量对QY的影响
df['Material_Count'] = df[['Material 1 (SMILES)', 'Material 2 (SMILES)', 'Material 3 (SMILES)']].notna().sum(axis=1)
material_qy_dict = {}
for count in sorted(df['Material_Count'].unique()):
    data = df[df['Material_Count'] == count]['Quantum yield(%)'].dropna().values
    if len(data) > 0:
        material_qy_dict[f'{int(count)}种材料'] = data
save_violin_data_and_plot(material_qy_dict, 'Material_Count', 'Quantum yield(%)', 
                          '不同材料数量对量子产率的影响', 
                          '小提琴图_量子产率_材料数量')

# 前驱体类型对QY的影响
def classify_precursor(smiles):
    if pd.isna(smiles):
        return 'None'
    smiles_str = str(smiles).lower()
    if 'op(=o)(o)o' in smiles_str:
        return 'Phosphorus'
    elif 'b(o)o' in smiles_str:
        return 'Boron'
    elif 's(=o)(=o)(o)o' in smiles_str:
        return 'Sulfur'
    elif any(x in smiles_str for x in ['c', 'n', 'o']):
        return 'Organic'
    return 'Other'

df['Precursor1_Type'] = df['Material 1 (SMILES)'].apply(classify_precursor)
precursor_qy_dict = {}
for ptype in df['Precursor1_Type'].unique():
    data = df[df['Precursor1_Type'] == ptype]['Quantum yield(%)'].dropna().values
    if len(data) > 0:
        precursor_qy_dict[ptype] = data
save_violin_data_and_plot(precursor_qy_dict, 'Precursor_Type', 'Quantum yield(%)', 
                          '前驱体类型对量子产率的影响', 
                          '小提琴图_量子产率_前驱体类型')

# 4. 发射波长的小提琴图
print("\n生成发射波长影响因素小提琴图... - 数据分布.py:312")

# 溶剂类型对Em的影响
solvent_em_dict = {}
for solvent in df['Solvent_Type'].unique():
    data = df[df['Solvent_Type'] == solvent]['Em(nm)'].dropna().values
    if len(data) > 0:
        solvent_em_dict[solvent] = data
save_violin_data_and_plot(solvent_em_dict, 'Solvent_Type', 'Em(nm)', 
                          '不同溶剂类型对发射波长的影响', 
                          '小提琴图_发射波长_溶剂类型')

# 温度对Em的影响
df_temp_em = df[['Temperature (℃)', 'Em(nm)']].dropna()
if len(df_temp_em) > 0:
    df_temp_em['Temp_Group'] = pd.cut(df_temp_em['Temperature (℃)'], bins=[0, 120, 160, 200, 240, 300], 
                                      labels=['<120', '120-160', '160-200', '200-240', '>240'])
    temp_em_dict = {}
    for temp_group in df_temp_em['Temp_Group'].unique():
        data = df_temp_em[df_temp_em['Temp_Group'] == temp_group]['Em(nm)'].values
        if len(data) > 0:
            temp_em_dict[str(temp_group)] = data
    save_violin_data_and_plot(temp_em_dict, 'Temperature_Group', 'Em(nm)', 
                              '不同温度对发射波长的影响', 
                              '小提琴图_发射波长_温度范围')

# 材料数量对Em的影响
material_em_dict = {}
for count in sorted(df['Material_Count'].unique()):
    data = df[df['Material_Count'] == count]['Em(nm)'].dropna().values
    if len(data) > 0:
        material_em_dict[f'{int(count)}种材料'] = data
save_violin_data_and_plot(material_em_dict, 'Material_Count', 'Em(nm)', 
                          '材料数量对发射波长的影响', 
                          '小提琴图_发射波长_材料数量')

# 5. 斯托克斯位移的小提琴图
print("\n生成斯托克斯位移影响因素小提琴图... - 数据分布.py:349")

# 溶剂类型对斯托克斯位移的影响
solvent_stoke_dict = {}
for solvent in df['Solvent_Type'].unique():
    data = df[df['Solvent_Type'] == solvent]['stoke (nm)'].dropna().values
    if len(data) > 0:
        solvent_stoke_dict[solvent] = data
save_violin_data_and_plot(solvent_stoke_dict, 'Solvent_Type', 'stoke (nm)', 
                          '溶剂类型对斯托克斯位移的影响', 
                          '小提琴图_斯托克斯位移_溶剂类型')

# 温度对斯托克斯位移的影响
df_temp_stoke = df[['Temperature (℃)', 'stoke (nm)']].dropna()
if len(df_temp_stoke) > 0:
    df_temp_stoke['Temp_Group'] = pd.cut(df_temp_stoke['Temperature (℃)'], bins=[0, 140, 180, 220, 300], 
                                         labels=['<140', '140-180', '180-220', '>220'])
    temp_stoke_dict = {}
    for temp_group in df_temp_stoke['Temp_Group'].unique():
        data = df_temp_stoke[df_temp_stoke['Temp_Group'] == temp_group]['stoke (nm)'].values
        if len(data) > 0:
            temp_stoke_dict[str(temp_group)] = data
    save_violin_data_and_plot(temp_stoke_dict, 'Temperature_Group', 'stoke (nm)', 
                              '温度对斯托克斯位移的影响', 
                              '小提琴图_斯托克斯位移_温度范围')

# 6. 粒径的小提琴图
print("\n生成粒径影响因素小提琴图... - 数据分布.py:376")

# 溶剂类型对粒径的影响
solvent_size_dict = {}
for solvent in df['Solvent_Type'].unique():
    data = df[df['Solvent_Type'] == solvent]['particle size(nm)'].dropna().values
    if len(data) > 0:
        solvent_size_dict[solvent] = data
save_violin_data_and_plot(solvent_size_dict, 'Solvent_Type', 'particle size(nm)', 
                          '溶剂类型对粒径的影响', 
                          '小提琴图_粒径_溶剂类型')

# 温度对粒径的影响
df_temp_size = df[['Temperature (℃)', 'particle size(nm)']].dropna()
if len(df_temp_size) > 0:
    df_temp_size['Temp_Group'] = pd.cut(df_temp_size['Temperature (℃)'], bins=[0, 140, 180, 220, 300], 
                                        labels=['<140', '140-180', '180-220', '>220'])
    temp_size_dict = {}
    for temp_group in df_temp_size['Temp_Group'].unique():
        data = df_temp_size[df_temp_size['Temp_Group'] == temp_group]['particle size(nm)'].values
        if len(data) > 0:
            temp_size_dict[str(temp_group)] = data
    save_violin_data_and_plot(temp_size_dict, 'Temperature_Group', 'particle size(nm)', 
                              '温度对粒径的影响', 
                              '小提琴图_粒径_温度范围')

# 7. 相关性分析
print("\n - 数据分布.py:403" + "="*80)
print("进行相关性分析... - 数据分布.py:404")
print("= - 数据分布.py:405"*80)

# 选择用于相关性分析的数值变量
corr_vars = ['Quantum yield(%)', 'Em(nm)', 'stoke (nm)', 'particle size(nm)', 
             'Temperature (℃)', 'Time (h)', 'volume (mL)', 'Ex(nm)', 'Abs(nm)',
             'Material 1 (M)', 'Material 2 (M)', 'Solvent_PC1', 'Solvent_PC2']

# 确保列存在
corr_vars = [var for var in corr_vars if var in df.columns]

# 计算相关性矩阵
corr_matrix = df[corr_vars].corr()

# 保存相关性矩阵到Excel
with pd.ExcelWriter('相关性热图数据.xlsx') as writer:
    corr_matrix.to_excel(writer, sheet_name='相关性矩阵')
    
    # 添加详细的相关性分析
    for target in ['Quantum yield(%)', 'Em(nm)', 'stoke (nm)']:
        if target in corr_matrix.columns:
            target_corr = corr_matrix[target].sort_values(ascending=False)
            target_df = pd.DataFrame({
                'Variable': target_corr.index,
                'Correlation': target_corr.values
            })
            target_df.to_excel(writer, sheet_name=f'{target}_相关性', index=False)

# 绘制相关性热图
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, fmt='.2f', 
            cbar_kws={"shrink": 0.8})
plt.title('变量间相关性热图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('相关性热图.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 8. 详细相关性分析输出
print("\n与量子产率的相关性分析: - 数据分布.py:445")
qy_corr = corr_matrix['Quantum yield(%)'].sort_values(ascending=False)
for var, corr_val in qy_corr.items():
    if var != 'Quantum yield(%)' and not np.isnan(corr_val):
        print(f"{var}: r = {corr_val:.3f} - 数据分布.py:449")

print("\n与发射波长的相关性分析: - 数据分布.py:451")
em_corr = corr_matrix['Em(nm)'].sort_values(ascending=False)
for var, corr_val in em_corr.items():
    if var != 'Em(nm)' and not np.isnan(corr_val):
        print(f"{var}: r = {corr_val:.3f} - 数据分布.py:455")

print("\n与斯托克斯位移的相关性分析: - 数据分布.py:457")
stoke_corr = corr_matrix['stoke (nm)'].sort_values(ascending=False)
for var, corr_val in stoke_corr.items():
    if var != 'stoke (nm)' and not np.isnan(corr_val):
        print(f"{var}: r = {corr_val:.3f} - 数据分布.py:461")

# 9. SMILES格式验证和保存
print("\n - 数据分布.py:464" + "="*80)
print("SMILES格式验证... - 数据分布.py:465")
print("= - 数据分布.py:466"*80)

def validate_smiles(smiles):
    """简单的SMILES格式验证"""
    if pd.isna(smiles):
        return 'Missing'
    smiles_str = str(smiles).strip()
    if len(smiles_str) == 0:
        return 'Empty'
    if any(c in smiles_str for c in ['C', 'c', 'N', 'n', 'O', 'o']):
        return 'Valid'
    return 'Invalid'

# 验证SMILES列并保存
smiles_cols = ['Material 1 (SMILES)', 'Material 2 (SMILES)', 'Material 3 (SMILES)']
smiles_analysis = []
for col in smiles_cols:
    if col in df.columns:
        df[f'{col}_Status'] = df[col].apply(validate_smiles)
        status_counts = df[f'{col}_Status'].value_counts()
        print(f"\n{col}: - 数据分布.py:486")
        for status, count in status_counts.items():
            print(f"{status}: {count} ({count/len(df)*100:.1f}%) - 数据分布.py:488")
            smiles_analysis.append({'Column': col, 'Status': status, 'Count': count, 'Percentage': count/len(df)*100})

# 保存SMILES分析结果
smiles_df = pd.DataFrame(smiles_analysis)
smiles_df.to_excel('SMILES格式验证结果.xlsx', index=False)

# 10. 保存所有处理后的数据
df.to_excel('processed_carbon_dots_data.xlsx', index=False)
print(f"\n处理后的完整数据已保存至: processed_carbon_dots_data.xlsx - 数据分布.py:497")

# 11. 生成数据文件清单
file_list = pd.DataFrame({
    '文件名': [
        '数据分布特征_量子产率_原始数据.xlsx',
        '数据分布特征_量子产率_直方图数据.xlsx',
        '数据分布特征_发射波长_原始数据.xlsx',
        '数据分布特征_发射波长_直方图数据.xlsx',
        '数据分布特征_斯托克斯位移_原始数据.xlsx',
        '数据分布特征_斯托克斯位移_直方图数据.xlsx',
        '数据分布特征_粒径_原始数据.xlsx',
        '数据分布特征_粒径_直方图数据.xlsx',
        '数据分布特征_温度_原始数据.xlsx',
        '数据分布特征_温度_直方图数据.xlsx',
        '数据分布特征_时间_原始数据.xlsx',
        '数据分布特征_时间_直方图数据.xlsx',
        '小提琴图_量子产率_溶剂类型.xlsx',
        '小提琴图_量子产率_温度范围.xlsx',
        '小提琴图_量子产率_反应时间.xlsx',
        '小提琴图_量子产率_材料数量.xlsx',
        '小提琴图_量子产率_前驱体类型.xlsx',
        '小提琴图_发射波长_溶剂类型.xlsx',
        '小提琴图_发射波长_温度范围.xlsx',
        '小提琴图_发射波长_材料数量.xlsx',
        '小提琴图_斯托克斯位移_溶剂类型.xlsx',
        '小提琴图_斯托克斯位移_温度范围.xlsx',
        '小提琴图_粒径_溶剂类型.xlsx',
        '小提琴图_粒径_温度范围.xlsx',
        '相关性热图数据.xlsx',
        'SMILES格式验证结果.xlsx',
        'processed_carbon_dots_data.xlsx'
    ],
    '描述': [
        '量子产率原始数据',
        '量子产率直方图统计',
        '发射波长原始数据',
        '发射波长直方图统计',
        '斯托克斯位移原始数据',
        '斯托克斯位移直方图统计',
        '粒径原始数据',
        '粒径直方图统计',
        '温度原始数据',
        '温度直方图统计',
        '时间原始数据',
        '时间直方图统计',
        '溶剂类型对QY影响数据',
        '温度范围对QY影响数据',
        '反应时间对QY影响数据',
        '材料数量对QY影响数据',
        '前驱体类型对QY影响数据',
        '溶剂类型对Em影响数据',
        '温度范围对Em影响数据',
        '材料数量对Em影响数据',
        '溶剂类型对斯托克斯位移影响数据',
        '温度范围对斯托克斯位移影响数据',
        '溶剂类型对粒径影响数据',
        '温度范围对粒径影响数据',
        '相关性矩阵和详细分析',
        'SMILES格式验证结果',
        '处理后的完整数据集'
    ]
})

file_list.to_excel('生成的数据文件清单.xlsx', index=False)
print(f"数据文件清单已保存至: 生成的数据文件清单.xlsx - 数据分布.py:562")

# 12. 生成分析报告
report = f"""
========================================
碳点实验数据分析报告
========================================

数据概况:
- 总样本数: {len(df)}
- 数值变量数: {len(corr_vars)}

量子产率 (QY):
- 范围: {df['Quantum yield(%)'].min():.2f}% - {df['Quantum yield(%)'].max():.2f}%
- 均值: {df['Quantum yield(%)'].mean():.2f}%
- 中位数: {df['Quantum yield(%)'].median():.2f}%
- 标准差: {df['Quantum yield(%)'].std():.2f}%

发射波长 (Em):
- 范围: {df['Em(nm)'].min():.0f} nm - {df['Em(nm)'].max():.0f} nm
- 均值: {df['Em(nm)'].mean():.0f} nm
- 中位数: {df['Em(nm)'].median():.0f} nm
- 标准差: {df['Em(nm)'].std():.0f} nm

斯托克斯位移:
- 范围: {df['stoke (nm)'].min():.0f} nm - {df['stoke (nm)'].max():.0f} nm
- 均值: {df['stoke (nm)'].mean():.0f} nm
- 中位数: {df['stoke (nm)'].median():.0f} nm
- 标准差: {df['stoke (nm)'].std():.0f} nm

粒径:
- 范围: {df['particle size(nm)'].min():.2f} nm - {df['particle size(nm)'].max():.2f} nm
- 均值: {df['particle size(nm)'].mean():.2f} nm
- 中位数: {df['particle size(nm)'].median():.2f} nm
- 标准差: {df['particle size(nm)'].std():.2f} nm

生成的文件:
柱状图相关文件 (带拟合曲线):
- 数据分布特征_量子产率.png (及对应的Excel数据)
- 数据分布特征_发射波长.png (及对应的Excel数据)
- 数据分布特征_斯托克斯位移.png (及对应的Excel数据)
- 数据分布特征_粒径.png (及对应的Excel数据)
- 数据分布特征_温度.png (及对应的Excel数据)
- 数据分布特征_时间.png (及对应的Excel数据)

小提琴图相关文件:
- 小提琴图_量子产率_溶剂类型.png (及对应的Excel数据)
- 小提琴图_量子产率_温度范围.png (及对应的Excel数据)
- 小提琴图_量子产率_反应时间.png (及对应的Excel数据)
- 小提琴图_量子产率_材料数量.png (及对应的Excel数据)
- 小提琴图_量子产率_前驱体类型.png (及对应的Excel数据)
- 小提琴图_发射波长_溶剂类型.png (及对应的Excel数据)
- 小提琴图_发射波长_温度范围.png (及对应的Excel数据)
- 小提琴图_发射波长_材料数量.png (及对应的Excel数据)
- 小提琴图_斯托克斯位移_溶剂类型.png (及对应的Excel数据)
- 小提琴图_斯托克斯位移_温度范围.png (及对应的Excel数据)
- 小提琴图_粒径_溶剂类型.png (及对应的Excel数据)
- 小提琴图_粒径_温度范围.png (及对应的Excel数据)

相关性分析:
- 相关性热图.png
- 相关性热图数据.xlsx

其他:
- SMILES格式验证结果.xlsx
- processed_carbon_dots_data.xlsx
- 生成的数据文件清单.xlsx

模型构建建议:
推荐使用以下特征进行预测建模:
1. 量子产率预测: 使用溶剂PC值、温度、时间、前驱体浓度
2. 发射波长预测: 使用溶剂类型、温度、激发波长
3. 斯托克斯位移预测: 使用发射波长、激发波长、溶剂性质
"""

with open('数据分析报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n数据分析报告已保存至: 数据分析报告.txt - 数据分布.py:640")
print(report)