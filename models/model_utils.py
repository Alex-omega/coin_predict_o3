import os
import logging
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, Input, Model

# 从配置文件中导入模型相关配置和存储路径
from config.settings import COIN_MODELS_DIR, MOTHER_MODEL_PARAMS, FINE_TUNE_PARAMS

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
# 函数：构建母模型
# --------------------------
def build_mother_model(input_shape, model_params=MOTHER_MODEL_PARAMS):
    """
    构建一个更复杂的母模型，采用混合CNN和LSTM的架构，
    用于捕捉时间序列数据的局部特征与长期依赖性。
    参数:
      - input_shape: 模型输入的形状，例如 (time_steps, feature_count)
      - model_params: 模型参数字典（包含学习率等设置）
    返回:
      - 编译后的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # 卷积层：提取局部时序特征
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # LSTM层：捕捉长期依赖性
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    
    # 全连接层：融合特征并输出预测值
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = optimizers.Adam(learning_rate=model_params.get("learning_rate", 0.001))
    model.compile(optimizer=optimizer, loss='mse')
    
    logger.info("复杂母模型构建完成")
    return model

# --------------------------
# 函数：训练母模型
# --------------------------
def train_mother_model(model, train_data, train_labels, model_params=MOTHER_MODEL_PARAMS):
    """
    使用给定的训练数据对母模型进行训练。
    参数:
      - model: 已构建并编译的Keras模型
      - train_data: 训练数据（输入）
      - train_labels: 训练标签（目标值）
      - model_params: 模型训练参数（例如epochs、batch_size等）
    返回:
      - 训练历史对象
    """
    epochs = model_params.get("epochs", 10)
    batch_size = model_params.get("batch_size", 512)
    
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)
    logger.info("复杂母模型训练完成")
    return history

# --------------------------
# 函数：高级币种模型微调（保留前述高级微调函数）
# --------------------------
def fine_tune_coin_model_advanced(mother_model, coin_train_data, coin_train_labels, fine_tune_params=FINE_TUNE_PARAMS, freeze_until=None):
    """
    基于母模型对单个币种进行更高级的微调，支持冻结部分层、早停和学习率调度。
    参数:
      - mother_model: 预训练的母模型
      - coin_train_data: 单个币种的训练数据
      - coin_train_labels: 单个币种的训练标签
      - fine_tune_params: 包含微调参数（如epochs、batch_size、学习率等）
      - freeze_until: 指定冻结的层索引（若为None则不冻结任何层）
    返回:
      - 微调后的币种模型和训练历史对象
    """
    # 克隆母模型结构和权重，避免直接修改母模型
    coin_model = tf.keras.models.clone_model(mother_model)
    coin_model.set_weights(mother_model.get_weights())
    
    # 根据freeze_until参数冻结部分层
    if freeze_until is not None:
        for idx, layer in enumerate(coin_model.layers):
            if idx < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
        logger.info(f"冻结前{freeze_until}层，后续层进行微调")
    else:
        for layer in coin_model.layers:
            layer.trainable = True
    
    # 设置优化器和初始学习率
    initial_lr = fine_tune_params.get("learning_rate", 0.0005)
    optimizer = optimizers.Adam(learning_rate=initial_lr)
    coin_model.compile(optimizer=optimizer, loss='mse')
    
    # 定义回调：早停和学习率调度
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)
    ]
    
    epochs = fine_tune_params.get("epochs", 5)
    batch_size = fine_tune_params.get("batch_size", 128)
    
    history = coin_model.fit(
        coin_train_data, coin_train_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("高级币种模型微调完成")
    return coin_model, history

# --------------------------
# 函数：保存模型到磁盘
# --------------------------
def save_model(model, model_name):
    """
    将模型保存为HDF5格式文件。
    参数:
      - model: 要保存的Keras模型
      - model_name: 模型文件名（不含扩展名）
    返回:
      - 模型保存的完整文件路径
    """
    save_path = os.path.join(COIN_MODELS_DIR, model_name + ".h5")
    try:
        model.save(save_path)
        logger.info(f"模型已保存至: {save_path}")
    except Exception as e:
        logger.error(f"模型保存失败: {e}")
    return save_path

# --------------------------
# 函数：从磁盘加载模型
# --------------------------
def load_model(model_path):
    """
    从指定路径加载Keras模型。
    参数:
      - model_path: 模型文件路径
    返回:
      - 加载后的Keras模型；若加载失败返回None
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"模型已从 {model_path} 加载")
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None

# --------------------------
# 示例：如何使用构建与训练母模型的函数
# --------------------------
if __name__ == "__main__":
    # 假设输入形状为 (60, 6) —— 例如60个时间步，每步6个特征
    input_shape = (60, 6)
    mother_model = build_mother_model(input_shape)
    mother_model.summary()

    # 以下为示例训练数据（请替换为实际数据）
    import numpy as np
    train_data = np.random.rand(1000, 60, 6)
    train_labels = np.random.rand(1000, 1)
    
    history = train_mother_model(mother_model, train_data, train_labels)
