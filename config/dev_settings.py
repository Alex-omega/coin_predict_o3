from .settings import *

# ==========================
# 🛠 开发环境专用配置
# ==========================

# 调试模式
DEBUG = True

# 日志等级
LOGGING_LEVEL = "DEBUG"

# 数据抓取相关配置
FETCH_COIN_COUNT = 100  # 在开发环境中，减少抓取的币种数量以加快测试速度
HISTORICAL_DATA_PERIOD = "1 week"  # 缩短数据抓取周期，减小数据量

# 模型训练相关配置
MOTHER_MODEL_PARAMS = {
    "epochs": 2,  # 减少训练轮数，加快开发测试速度
    "batch_size": 128,  # 减小批量大小，降低资源占用
    "learning_rate": 0.01,  # 增大学习率，加快收敛
    "hidden_layers": [64, 32],  # 减少隐藏层神经元数量，简化模型
}

FINE_TUNE_PARAMS = {
    "epochs": 1,  # 减少微调训练轮数
    "learning_rate": 0.005,  # 增大学习率，加快微调速度
}

# 预测相关配置
PREDICTION_WINDOW = 2  # 缩短预测窗口，便于快速验证
DEV_MODE = True  # 启用开发模式，进行预测后等待并验证结果

# 数据存储路径
DEV_DATA_DIR = os.path.join(BASE_DIR, "../dev_data")  # 开发环境数据目录
RAW_DATA_DIR = os.path.join(DEV_DATA_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(DEV_DATA_DIR, "processed_data")
COIN_MODELS_DIR = os.path.join(DEV_DATA_DIR, "coin_models")
PREDICTION_RESULTS_DIR = os.path.join(DEV_DATA_DIR, "../predictions/prediction_results")
VISUALIZATIONS_DIR = os.path.join(DEV_DATA_DIR, "../predictions/visualizations")

# 确保开发环境目录存在
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, COIN_MODELS_DIR, PREDICTION_RESULTS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)
