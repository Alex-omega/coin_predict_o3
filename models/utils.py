import requests
import time
import logging
import hmac
import hashlib
from urllib.parse import urlencode

# 从配置文件中导入币安API相关配置
from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_BASE_URL

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
# 内部函数：发送GET请求
# --------------------------
def _get_request(url, params=None):
    """
    发送GET请求并返回JSON数据。
    参数:
      - url: 请求的完整URL地址
      - params: URL参数（字典形式）
    返回:
      - 请求成功时返回解析后的JSON数据，否则返回None
    """
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"GET请求失败: {e}")
        return None

# --------------------------
# 内部函数：发送POST请求
# --------------------------
def _post_request(url, data=None):
    """
    发送POST请求并返回JSON数据。
    参数:
      - url: 请求的完整URL地址
      - data: 发送的数据（字典形式）
    返回:
      - 请求成功时返回解析后的JSON数据，否则返回None
    """
    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"POST请求失败: {e}")
        return None

# --------------------------
# 签名函数：对请求参数进行签名
# --------------------------
def sign_request(params, secret=BINANCE_API_SECRET):
    """
    对请求参数进行HMAC SHA256签名，以满足币安API的安全要求。
    参数:
      - params: 请求参数（字典形式）
      - secret: API密钥的Secret部分（默认从配置中读取）
    返回:
      - 签名字符串
    """
    query_string = urlencode(params)
    signature = hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

# --------------------------
# 获取币安服务器时间
# --------------------------
def get_server_time():
    """
    获取币安服务器当前时间（毫秒）。
    返回:
      - 成功时返回服务器时间（毫秒），否则返回None
    """
    url = BINANCE_BASE_URL + "/api/v3/time"
    data = _get_request(url)
    if data and "serverTime" in data:
        return data["serverTime"]
    else:
        logger.error("无法获取币安服务器时间")
        return None

# --------------------------
# 获取币安历史K线数据
# --------------------------
def fetch_historical_klines(symbol, interval, start_time, end_time=None, limit=1000):
    """
    获取币安历史K线数据。
    参数:
      - symbol: 交易对符号，例如 'BTCUSDT'
      - interval: K线周期，如 '1m', '5m', '1h' 等
      - start_time: 起始时间戳（毫秒）
      - end_time: 结束时间戳（毫秒），默认为None时获取最新数据
      - limit: 单次返回的最大数据条数（默认1000，币安接口限制为1000）
    返回:
      - 历史K线数据（列表形式），每个元素为一根K线的列表数据
    """
    url = BINANCE_BASE_URL + "/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit
    }
    if end_time:
        params["endTime"] = end_time

    data = _get_request(url, params=params)
    return data

# --------------------------
# 获取币安交易所信息
# --------------------------
def fetch_exchange_info():
    """
    获取币安交易所信息，包括所有交易对和相关规则。
    返回:
      - 交易所信息的JSON数据
    """
    url = BINANCE_BASE_URL + "/api/v3/exchangeInfo"
    data = _get_request(url)
    return data

# --------------------------
# 时间转换工具函数
# --------------------------
def timestamp_to_str(timestamp):
    """
    将时间戳（毫秒）转换为格式化的字符串（"YYYY-MM-DD HH:MM:SS"）。
    参数:
      - timestamp: 时间戳，单位为毫秒
    返回:
      - 格式化的时间字符串
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp / 1000))

def str_to_timestamp(time_str, time_format="%Y-%m-%d %H:%M:%S"):
    """
    将时间字符串转换为时间戳（毫秒）。
    参数:
      - time_str: 时间字符串
      - time_format: 时间字符串的格式（默认"YYYY-MM-DD HH:MM:SS"）
    返回:
      - 时间戳，单位为毫秒（整数形式）
    """
    return int(time.mktime(time.strptime(time_str, time_format)) * 1000)
