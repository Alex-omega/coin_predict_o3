import os
import time
import logging
import argparse
import numpy as np
import pandas as pd

# 导入项目配置
from config import settings

# 导入各模块函数
from models import data_preprocessing, model_utils
from models.utils import fetch_historical_klines  # 若后续扩展实际调用API

# --------------------------
# 设置日志记录器
# --------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# 函数：数据抓取（模拟示例）
# --------------------------
def fetch_data():
    """
    模拟从币安API抓取数据，将样例数据保存到 RAW_DATA_DIR 目录中。
    实际使用中，可调用币安API抓取每组币的1分钟级别历史数据。
    """
    logging.info("开始抓取数据...")
    # 模拟数据：生成过去一个月（约43200分钟）的样例数据
    num_rows = 43200  # 1个月1分钟数据
    timestamps = np.arange(num_rows) * 60000  # 每分钟对应的毫秒时间戳
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.random.rand(num_rows),
        "high": np.random.rand(num_rows),
        "low": np.random.rand(num_rows),
        "close": np.random.rand(num_rows),
        "volume": np.random.rand(num_rows)
    })
    # 保存到 RAW_DATA_DIR
    sample_data_path = os.path.join(settings.RAW_DATA_DIR, "sample_data.csv")
    df.to_csv(sample_data_path, index=False)
    logging.info(f"样例数据已保存至 {sample_data_path}")

# --------------------------
# 函数：训练母模型
# --------------------------
def train_mother():
    """
    预处理抓取的数据，构造训练样本，并训练母模型。
    训练完成后将母模型保存到 COIN_MODELS_DIR 目录中。
    """
    sample_data_path = os.path.join(settings.RAW_DATA_DIR, "sample_data.csv")
    # 预处理数据：清洗、添加技术指标、归一化
    df = data_preprocessing.process_data(sample_data_path, save_processed=True)
    if df is None:
        logging.error("数据预处理失败，无法训练母模型")
        return

    # 构造滑动窗口数据，示例中采用最后60个时间步作为输入
    timesteps = 60
    feature_cols = ['close', 'ma5', 'ma15', 'ma30', 'volatility', 'rsi']
    data = df[feature_cols].values
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps][0])  # 使用close作为预测目标
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # 构建母模型并训练
    input_shape = (X.shape[1], X.shape[2])
    mother_model = model_utils.build_mother_model(input_shape)
    history = model_utils.train_mother_model(mother_model, X, y)
    
    # 保存母模型
    model_utils.save_model(mother_model, "mother_model")
    logging.info("母模型训练并保存完成")

# --------------------------
# 函数：币种模型微调与预测
# --------------------------
def fine_tune_and_predict():
    """
    基于预处理数据，从母模型加载或重新训练后，
    对单个币种进行微调，并预测下一时刻的走势。
    """
    sample_data_path = os.path.join(settings.RAW_DATA_DIR, "sample_data.csv")
    df = data_preprocessing.process_data(sample_data_path, save_processed=False)
    if df is None:
        logging.error("数据预处理失败，无法进行币种微调")
        return

    feature_cols = ['close', 'ma5', 'ma15', 'ma30', 'volatility', 'rsi']
    data = df[feature_cols].values
    timesteps = 60
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps][0])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # 使用部分数据作为币种专属训练数据（示例中取前100个样本）
    coin_train_data = X[:100]
    coin_train_labels = y[:100]
    
    # 尝试加载已有的母模型，如不存在则重新训练
    mother_model_path = os.path.join(settings.COIN_MODELS_DIR, "mother_model.h5")
    mother_model = model_utils.load_model(mother_model_path)
    if mother_model is None:
        logging.info("未找到母模型，重新构建并训练母模型")
        input_shape = (X.shape[1], X.shape[2])
        mother_model = model_utils.build_mother_model(input_shape)
        model_utils.train_mother_model(mother_model, X, y)
    
    # 使用高级微调策略对币种模型进行微调，冻结前2层作为示例
    coin_model, history = model_utils.fine_tune_coin_model_advanced(mother_model, coin_train_data, coin_train_labels, freeze_until=2)
    
    # 保存微调后的币种模型
    model_utils.save_model(coin_model, "coin_model_sample")
    
    # 使用最后一个时间窗口进行预测
    sample_input = X[-1].reshape(1, X.shape[1], X.shape[2])
    prediction = coin_model.predict(sample_input)
    logging.info(f"币种模型预测下一时刻的价格为: {prediction[0][0]}")

# --------------------------
# 函数：预测对比（开发模式下使用）
# --------------------------
def compare_predictions():
    """
    在开发模式下，等待5分钟后抓取最新数据，
    与先前预测结果进行对比，并记录误差。
    """
    logging.info("开发模式：等待5分钟以获取新数据进行预测对比...")
    time.sleep(5 * 60)  # 暂停5分钟
    
    # 模拟抓取最新数据（实际应调用数据抓取接口）
    sample_data_path = os.path.join(settings.RAW_DATA_DIR, "sample_data.csv")
    df_new = data_preprocessing.process_data(sample_data_path, save_processed=False)
    if df_new is None:
        logging.error("新数据预处理失败，无法进行对比")
        return

    feature_cols = ['close', 'ma5', 'ma15', 'ma30', 'volatility', 'rsi']
    data_new = df_new[feature_cols].values
    timesteps = 60
    X_new = []
    for i in range(len(data_new) - timesteps):
        X_new.append(data_new[i:i+timesteps])
    X_new = np.array(X_new)
    
    # 加载之前微调的币种模型
    coin_model_path = os.path.join(settings.COIN_MODELS_DIR, "coin_model_sample.h5")
    coin_model = model_utils.load_model(coin_model_path)
    if coin_model is None:
        logging.error("币种模型加载失败，无法进行预测对比")
        return
    
    # 使用最后一个窗口数据进行预测，并与实际close值对比
    sample_input = X_new[-1].reshape(1, X_new.shape[1], X_new.shape[2])
    prediction = coin_model.predict(sample_input)
    actual_value = data_new[len(data_new) - 1][0]  # 假设最后一个数据的close值
    logging.info(f"对比结果 - 预测值: {prediction[0][0]}, 实际值: {actual_value}")

# --------------------------
# 主函数：解析命令行参数并执行对应任务
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="超短线币圈投资辅助系统主入口")
    parser.add_argument("action", type=str, choices=["fetch", "train", "finetune", "predict", "compare"],
                        help="指定要执行的任务: fetch（抓取数据）, train（训练母模型）, finetune（币种微调并预测）, compare（预测对比）")
    args = parser.parse_args()
    
    if args.action == "fetch":
        fetch_data()
    elif args.action == "train":
        train_mother()
    elif args.action in ["finetune", "predict"]:
        fine_tune_and_predict()
    elif args.action == "compare":
        compare_predictions()
    else:
        logging.error("未知的操作")

if __name__ == "__main__":
    main()
