import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 關閉 GPU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 你的花朵資料夾路徑
data_dir = "flower"

# 參數設定
img_size = (180, 180)
batch_size = 32

# 建立訓練/驗證資料集
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

# 標準化
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 類別數量
num_classes = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# 簡單 CNN 模型
model = keras.Sequential([
    layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=img_size + (3,)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 訓練模型
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# 測試：拿一張圖片做預測
img_path = "flower/roses/xxx.jpg"  # 換成你的測試圖片
img = keras.utils.load_img(img_path, target_size=img_size)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with {:.2f}% confidence."
      .format(train_ds.class_names[tf.argmax(score)], 100 * tf.reduce_max(score)))
