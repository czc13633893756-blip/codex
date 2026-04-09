import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
import matplotlib
from scipy.signal import savgol_filter
from scipy import linalg

# 修复matplotlib后端问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import traceback
import os

warnings.filterwarnings('ignore')


class SpectrumPreprocessor:
    """光谱数据预处理类"""

    @staticmethod
    def sg_smoothing(X, window_length=11, polyorder=2):
        """Savitzky-Golay平滑"""
        return savgol_filter(X, window_length, polyorder, axis=1)

    @staticmethod
    def first_derivative(X):
        """一阶导数"""
        return np.gradient(X, axis=1)

    @staticmethod
    def snv(X):
        """标准正态变量变换"""
        X_snv = np.zeros_like(X)
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            mean_val = np.mean(spectrum)
            std_val = np.std(spectrum)
            if std_val > 0:
                X_snv[i, :] = (spectrum - mean_val) / std_val
            else:
                X_snv[i, :] = spectrum - mean_val
        return X_snv

    @staticmethod
    def msc(X):
        """多元散射校正"""
        mean_spectrum = np.mean(X, axis=0)
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            A = np.vstack([mean_spectrum, np.ones(len(mean_spectrum))]).T
            try:
                result = linalg.lstsq(A, X[i, :])
                slope, intercept = result[0]
            except TypeError:
                result = linalg.lstsq(A, X[i, :])
                slope, intercept = result[0]
            if abs(slope) > 1e-10:
                X_msc[i, :] = (X[i, :] - intercept) / slope
            else:
                X_msc[i, :] = X[i, :] - intercept
        return X_msc

    @staticmethod
    def snv_d1st(X):
        """SNV + 一阶导数"""
        X_snv = SpectrumPreprocessor.snv(X)
        return SpectrumPreprocessor.first_derivative(X_snv)

    @staticmethod
    def apply_preprocessing(X, method='snv_d1st'):
        """应用指定的预处理方法"""
        preprocess_methods = {
            'snv_d1st': SpectrumPreprocessor.snv_d1st
        }

        if method not in preprocess_methods:
            raise ValueError(f"不支持的预处理方法: {method}")

        print(f"应用预处理方法: {method}")
        return preprocess_methods[method](X)


class DataLoader:
    """数据加载器"""

    @staticmethod
    def load_spectrum_data(file_path):
        """加载光谱数据"""
        print(f"加载光谱数据: {file_path}")
        spectrum_data = pd.read_excel(file_path, header=0)
        print(f"光谱数据形状: {spectrum_data.shape}")

        variety_data = spectrum_data.iloc[:, 0].values
        maturity_data = spectrum_data.iloc[:, 1].values
        sample_ids = spectrum_data.iloc[:, 2].values
        X = spectrum_data.iloc[:, 3:].values
        wavelength_names = list(spectrum_data.columns[3:])

        return X, sample_ids, wavelength_names, variety_data, maturity_data

    @staticmethod
    def load_catechin_data(file_path):
        """加载儿茶素数据"""
        print(f"加载儿茶素数据: {file_path}")
        catechin_data = pd.read_excel(file_path, header=0)
        print(f"儿茶素数据形状: {catechin_data.shape}")

        variety_data = catechin_data.iloc[:, 0].values
        maturity_data = catechin_data.iloc[:, 1].values
        sample_ids = catechin_data.iloc[:, 2].values
        catechin_names = list(catechin_data.columns[3:])
        y = catechin_data.iloc[:, 3:].values

        return y, sample_ids, catechin_names, variety_data, maturity_data

    @staticmethod
    def align_data_by_catechin(spectrum_ids, spectrum_X, spectrum_variety, spectrum_maturity,
                               catechin_ids, catechin_y, catechin_variety, catechin_maturity):
        """以儿茶素数据为基准对齐数据"""
        print("\n以儿茶素数据为基准进行数据对齐...")

        spectrum_dict = {}
        for i, geti in enumerate(spectrum_ids):
            if geti not in spectrum_dict:
                spectrum_dict[geti] = {
                    'index': i,
                    'X': spectrum_X[i],
                    'variety': spectrum_variety[i],
                    'maturity': spectrum_maturity[i]
                }

        aligned_X = []
        aligned_y = []
        aligned_ids = []
        aligned_variety = []
        aligned_maturity = []
        missing_count = 0

        for i, geti in enumerate(catechin_ids):
            if geti in spectrum_dict:
                spectrum_info = spectrum_dict[geti]

                # 简单验证
                if catechin_variety[i] != spectrum_info['variety']:
                    pass  # 这里暂时跳过严格报错，防止数据细微拼写差异导致中断

                aligned_X.append(spectrum_info['X'])
                aligned_y.append(catechin_y[i])
                aligned_ids.append(geti)
                aligned_variety.append(catechin_variety[i])
                aligned_maturity.append(catechin_maturity[i])
            else:
                missing_count += 1

        aligned_X = np.array(aligned_X)
        aligned_y = np.array(aligned_y)
        aligned_ids = np.array(aligned_ids)
        aligned_variety = np.array(aligned_variety)
        aligned_maturity = np.array(aligned_maturity)

        print(f"成功对齐的样本数: {len(aligned_ids)}")
        if len(aligned_ids) == 0:
            raise ValueError("没有找到共同的样本ID，请检查数据文件！")

        return aligned_X, aligned_y, aligned_ids, aligned_variety, aligned_maturity


class AutoPLSR:
    """
    自动优化的偏最小二乘回归 (PLSR) 模型
    功能：
    1. 自动标准化数据 (StandardScaler)
    2. 使用交叉验证 (CV) 自动选择最佳主成分数 (n_components)
    """

    def __init__(self, max_components=20, cv_splits=5):
        self.max_components = max_components
        self.cv_splits = cv_splits
        self.model = None
        self.optimal_components = 0
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.feature_importances_ = None  # 用于存储回归系数作为重要性

    def fit(self, X, y):
        """训练 PLSR 模型"""
        # 1. 数据标准化
        X_scaled = self.X_scaler.fit_transform(X)
        # y 需要 reshape 为 (-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y_scaled = self.y_scaler.fit_transform(y)

        # 2. 自动寻找最佳主成分数
        # 限制最大成分数不能超过样本数或特征数
        max_comp = min(self.max_components, X.shape[0], X.shape[1])
        rmse_cv = []
        components_range = range(1, max_comp + 1)

        print(f"  正在通过 {self.cv_splits} 折交叉验证寻找最佳主成分数...", end='', flush=True)

        for i in components_range:
            # 临时模型，scale=False因为我们已经手动标准化了
            pls = PLSRegression(n_components=i, scale=False)
            # 使用 cross_val_predict 快速获取CV预测值
            y_cv = cross_val_predict(pls, X_scaled, y_scaled, cv=self.cv_splits)
            rmse = np.sqrt(mean_squared_error(y_scaled, y_cv))
            rmse_cv.append(rmse)

        # 选择 RMSE 最小对应的成分数
        self.optimal_components = np.argmin(rmse_cv) + 1
        print(f" 完成。最佳 LVs: {self.optimal_components}")

        # 3. 使用最佳成分数训练最终模型
        self.model = PLSRegression(n_components=self.optimal_components, scale=False)
        self.model.fit(X_scaled, y_scaled)

        # 存储回归系数（对于光谱分析很重要）
        self.feature_importances_ = self.model.coef_

    def predict(self, X_test):
        """预测"""
        # 标准化输入
        X_test_scaled = self.X_scaler.transform(X_test)
        # 预测（得到的是标准化后的结果）
        predictions_scaled = self.model.predict(X_test_scaled)
        # 反标准化还原为真实值
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        return predictions


def calculate_rpd(y_true, y_pred):
    """计算RPD（相对分析误差）指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_dev = np.std(y_true)

    if rmse == 0:
        return float('inf')

    rpd = std_dev / rmse
    return rpd


def evaluate_single_catechin(y_true, y_pred, catechin_name, dataset_name="测试集"):
    """评估单个儿茶素物质的回归模型性能"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    metrics = {}
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['evs'] = explained_variance_score(y_true, y_pred)
    metrics['rpd'] = calculate_rpd(y_true, y_pred)

    if metrics['rpd'] > 2.5:
        rpd_level = "优秀"
    elif metrics['rpd'] > 2.0:
        rpd_level = "良好"
    elif metrics['rpd'] > 1.8:
        rpd_level = "可接受"
    else:
        rpd_level = "需要改进"

    print(f"\n{catechin_name} - {dataset_name}评估结果:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  RPD: {metrics['rpd']:.6f} ({rpd_level})")

    return metrics


def plot_single_catechin_results(y_true, y_pred, catechin_name, dataset_name="测试集"):
    """绘制单个儿茶素物质的回归结果图"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{catechin_name} - {dataset_name}回归分析 (PLSR)', fontsize=16, fontweight='bold')

    # 1. 真实值 vs 预测值散点图
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6, color='blue')

    coeffs = np.polyfit(y_true, y_pred, 1)
    poly = np.poly1d(coeffs)
    y_fit = poly(y_true)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想线')
    ax.plot(y_true, y_fit, 'g-', alpha=0.8, label='回归线')

    ax.set_xlabel('真实值 (mg/g)')
    ax.set_ylabel('预测值 (mg/g)')
    ax.set_title('真实值 vs 预测值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 残差图
    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.6, color='green')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('预测值 (mg/g)')
    ax.set_ylabel('残差 (mg/g)')
    ax.set_title('残差图')
    ax.grid(True, alpha=0.3)

    # 3. 误差分布直方图
    ax = axes[1, 0]
    ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('残差 (mg/g)')
    ax.set_ylabel('频数')
    ax.set_title('残差分布')
    ax.grid(True, alpha=0.3)

    # 4. 指标汇总
    ax = axes[1, 1]
    ax.axis('off')

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rpd = calculate_rpd(y_true, y_pred)

    metrics_text = f"评估指标汇总 (PLSR):\n\n"
    metrics_text += f"MSE: {mse:.6f}\n"
    metrics_text += f"RMSE: {rmse:.6f}\n"
    metrics_text += f"MAE: {mae:.6f}\n"
    metrics_text += f"R²: {r2:.6f}\n"
    metrics_text += f"RPD: {rpd:.6f}\n\n"

    if rpd > 2.5:
        rpd_level = "优秀 (RPD > 2.5)"
    elif rpd > 2.0:
        rpd_level = "良好 (2.0 < RPD ≤ 2.5)"
    elif rpd > 1.8:
        rpd_level = "可接受 (1.8 < RPD ≤ 2.0)"
    else:
        rpd_level = "需要改进 (RPD ≤ 1.8)"

    metrics_text += f"RPD等级: {rpd_level}"

    ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    filename = f'{catechin_name}_{dataset_name}_results.png'.replace(' ', '_').replace('/', '_')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"图表已保存为: {filename}")
    return fig


def create_stratified_split(X, variety, maturity, test_size=0.2, random_state=42):
    """创建基于品种和成熟度的分层抽样划分"""
    stratify_labels = [f"{v}_{m}" for v, m in zip(variety, maturity)]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, stratify_labels):
        return train_index, test_index


def train_single_catechin_model(X, y, catechin_index, catechin_name, variety, maturity,
                                train_idx, test_idx, test_size=0.25, random_state=42):
    """训练单个儿茶素物质的 PLSR 模型"""
    print(f"\n{'=' * 60}")
    print(f"训练 {catechin_name} 模型 (PLSR)")
    print(f"{'=' * 60}")

    y_single = y[:, catechin_index].reshape(-1, 1)

    # 应用SNV+D1st预处理
    X_processed = SpectrumPreprocessor.apply_preprocessing(X, 'snv_d1st')

    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    y_train, y_test = y_single[train_idx], y_single[test_idx]

    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # =========================================================
    # 替换部分：使用 AutoPLSR 替代 BLSRegression
    # =========================================================
    # 设置最大主成分数为 20 (或样本数、波长数的最小值)
    # cv_splits=5 表示使用 5折交叉验证来寻找最佳参数
    model = AutoPLSR(max_components=20, cv_splits=5)

    start_time = datetime.datetime.now()
    model.fit(X_train, y_train)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"模型训练时间: {training_time:.2f}秒")

    # 预测
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_single_catechin(y_train, y_train_pred, catechin_name, "训练集")

    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_single_catechin(y_test, y_test_pred, catechin_name, "测试集")

    plot_single_catechin_results(y_train, y_train_pred, catechin_name, "训练集")
    plot_single_catechin_results(y_test, y_test_pred, catechin_name, "测试集")

    return {
        'catechin_name': catechin_name,
        'catechin_index': catechin_index,
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_idx': train_idx,
        'test_idx': test_idx
    }


def main():
    """主函数"""
    # 请修改这里的文件路径
    spectrum_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\儿茶素对应光谱 - 副本 - 副本.xlsx"
    catechin_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\数据-儿茶素-毫克每克 - 副本 - 副本.xlsx"

    print("=" * 60)
    print("茶叶儿茶素含量预测模型 (PLSR)")
    print("预处理: SNV + 一阶导数")
    print("=" * 60)

    try:
        X, spectrum_ids, wavelength_names, spectrum_variety, spectrum_maturity = DataLoader.load_spectrum_data(
            spectrum_path)
        y, catechin_ids, catechin_names, catechin_variety, catechin_maturity = DataLoader.load_catechin_data(
            catechin_path)

        X_aligned, y_aligned, aligned_ids, aligned_variety, aligned_maturity = DataLoader.align_data_by_catechin(
            spectrum_ids, X, spectrum_variety, spectrum_maturity,
            catechin_ids, y, catechin_variety, catechin_maturity)

        # 分层划分
        train_idx, test_idx = create_stratified_split(
            X_aligned, aligned_variety, aligned_maturity,
            test_size=0.2, random_state=42
        )

        all_results = []
        for i, catechin_name in enumerate(catechin_names):
            result = train_single_catechin_model(
                X_aligned, y_aligned, i, catechin_name,
                aligned_variety, aligned_maturity,
                train_idx, test_idx,
                test_size=0.2, random_state=42
            )
            all_results.append(result)

        # 汇总与保存
        summary_data = []
        for result in all_results:
            summary_data.append({
                '儿茶素物质': result['catechin_name'],
                '训练集R²': result['train_metrics']['r2'],
                '测试集R²': result['test_metrics']['r2'],
                '训练集RMSE': result['train_metrics']['rmse'],
                '测试集RMSE': result['test_metrics']['rmse'],
                '测试集RPD': result['test_metrics']['rpd'],
                '最佳LVs': result['model'].optimal_components  # 记录最佳主成分数
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('测试集R²', ascending=False)

        print("\n模型性能汇总表 (PLSR):")
        print(summary_df.to_string(index=False))

        output_dir = "plsr_catechin_results"
        os.makedirs(output_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False, encoding='utf-8-sig')

        # 简单保存预测详情 (可以根据需要恢复之前的完整保存逻辑)
        print(f"\n结果已保存至 {output_dir} 文件夹")

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()