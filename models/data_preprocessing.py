import os
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime

# 从配置文件中导入处理后数据的存储目录
from config.settings import PROCESSED_DATA_DIR

# --------------------------
# 设置日志记录器
# --------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --------------------------
# 函数：加载原始数据
# --------------------------
def load_raw_data(file_path):
    """
    从CSV文件加载原始数据。
    参数:
      - file_path: 原始数据文件路径
    返回:
      - pandas DataFrame 格式的数据，加载失败返回None
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"成功加载数据: {file_path}")
        return df
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None

# --------------------------
# 函数：数据清洗
# --------------------------
def clean_data(df):
    """
    对数据进行清洗，包括转换时间戳和删除缺失值。
    参数:
      - df: 原始数据的DataFrame
    返回:
      - 清洗后的DataFrame
    """
    # 如果数据中包含'timestamp'字段，则转换为datetime格式
    if 'timestamp' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        except Exception as e:
            logger.error(f"时间戳转换失败: {e}")
        # 删除原始时间戳列
        df.drop(columns=['timestamp'], inplace=True)
    
    # 删除存在缺失值的行
    df.dropna(inplace=True)
    return df

# --------------------------
# 函数：添加技术指标
# --------------------------
def add_technical_indicators(df):
    """
    添加常用的技术指标，如移动平均线、波动率和RSI。
    参数:
      - df: 清洗后的DataFrame，要求包含'close'列（收盘价）
    返回:
      - 添加了技术指标的DataFrame
    """
    # 计算5分钟、15分钟和30分钟移动平均线（假设数据频率为1分钟）
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    
    # 计算价格波动率（基于15分钟的标准差）
    df['volatility'] = df['close'].rolling(window=15).std()
    
    # 计算相对强弱指数 (RSI)
    delta = df['close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    gain_mean = gain.rolling(window=14).mean()
    loss_mean = loss.abs().rolling(window=14).mean()
    rs = gain_mean / (loss_mean + 1e-8)  # 避免除以0
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

# --------------------------
# 函数：归一化特征
# --------------------------
def normalize_features(df, feature_cols):
    """
    对指定的特征列进行Min-Max归一化处理。
    参数:
      - df: DataFrame数据
      - feature_cols: 需要归一化的列名列表
    返回:
      - 归一化后的DataFrame
    """
    df_normalized = df.copy()
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        # 进行归一化计算，若max==min则保持原值
        if max_val - min_val != 0:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = df[col]
    return df_normalized

# --------------------------
# 函数：综合数据处理流程
# --------------------------
def process_data(file_path, save_processed=True):
    """
    综合数据处理流程：
      1. 加载原始数据
      2. 数据清洗（时间戳转换、缺失值处理）
      3. 添加技术指标（移动平均、波动率、RSI等）
      4. 对关键特征归一化
      5. 可选：保存处理后的数据
    参数:
      - file_path: 原始数据文件路径
      - save_processed: 是否将处理后的数据保存到磁盘（默认True）
    返回:
      - 处理后的DataFrame，处理失败返回None
    """
    df = load_raw_data(file_path)
    if df is None:
        logger.error("数据加载失败，停止处理流程")
        return None

    df = clean_data(df)
    df = add_technical_indicators(df)
    
    # 选择需要归一化的特征列（可根据实际情况调整）
    feature_cols = ['close', 'ma5', 'ma15', 'ma30', 'volatility', 'rsi']
    df = normalize_features(df, feature_cols)
    
    if save_processed:
        # 根据原文件名生成处理后数据文件名
        filename = os.path.basename(file_path).replace(".csv", "_processed.csv")
        save_path = os.path.join(PROCESSED_DATA_DIR, filename)
        try:
            df.to_csv(save_path, index=False)
            logger.info(f"处理后的数据已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存处理数据失败: {e}")
    return df

# --------------------------
# 主函数：用于独立测试数据处理流程
# --------------------------
if __name__ == "__main__":
    # 示例文件路径（请确保存在相应的样例数据文件）
    test_file = os.path.join(os.path.dirname(__file__), "../data/raw_data/sample_data.csv")
    processed_df = process_data(test_file)
    if processed_df is not None:
        print("数据处理成功，预览处理后的数据:")
        print(processed_df.head())
