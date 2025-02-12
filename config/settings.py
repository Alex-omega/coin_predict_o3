import os

# ==========================
# 🔧 项目全局配置
# ==========================

# 数据存储路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
DATA_DIR = os.path.join(BASE_DIR, "../data")  # 主要数据目录
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")  # 存放原始抓取的数据
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")  # 处理后的数据
COIN_MODELS_DIR = os.path.join(DATA_DIR, "coin_models")  # 存放每个币种微调后的模型
PREDICTION_RESULTS_DIR = os.path.join(DATA_DIR, "../predictions/prediction_results")  # 存放预测结果
VISUALIZATIONS_DIR = os.path.join(DATA_DIR, "../predictions/visualizations")  # 存放可视化数据

# 确保目录存在
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, COIN_MODELS_DIR, PREDICTION_RESULTS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==========================
# 🔑 API 配置
# ==========================
BINANCE_API_KEY = "your_api_key_here"  # 币安 API Key（建议存入环境变量）
BINANCE_API_SECRET = "your_api_secret_here"  # 币安 API Secret（建议存入环境变量）
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# ==========================
# 📊 数据抓取相关配置
# ==========================
COIN_FILTER_CRITERIA = {
    "min_age_days": 7,  # 过滤掉存活时间小于7天的币
    "exclude_keywords": ["Trump", "Elon", "Meme"],  # 排除受外部因素影响过大的币
    "chain": "SOL",  # 仅关注 Solana 链上的币种
}

FETCH_COIN_COUNT = 2000  # 每周抓取 2000 组币的数据
HISTORICAL_DATA_INTERVAL = "1m"  # 1分钟 K 线数据
HISTORICAL_DATA_PERIOD = "1 month"  # 抓取过去一个月的数据

# ==========================
# 🤖 模型训练相关配置
# ==========================
MOTHER_MODEL_PARAMS = {
    "epochs": 10,  # 母模型训练轮数
    "batch_size": 512,  # 批量大小
    "learning_rate": 0.001,  # 学习率
    "hidden_layers": [256, 128, 64],  # 隐藏层神经元数
}

FINE_TUNE_PARAMS = {
    "epochs": 5,  # 每个币种微调训练轮数
    "learning_rate": 0.0005,  # 微调的学习率
}

# ==========================
# 🔮 预测相关配置
# ==========================
PREDICTION_WINDOW = 5  # 预测未来 5 分钟的走势
DEV_MODE = False  # 是否启用开发模式（启用后，预测后等待 5 分钟并获取实际数据做对比）

# ==========================
# 🛠 其他配置
# ==========================
LOGGING_LEVEL = "INFO"  # 日志等级，可选 "DEBUG"、"INFO"、"WARNING"、"ERROR"
