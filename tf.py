import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 關閉 GPU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# 1. 資料夾設定
# ----------------------------
data_dir = "flowers"   # 本地花朵資料夾
img_size = (224, 224)  # 輸入模型大小，可調大一點提升準確度
batch_size = 32

# ----------------------------
# 2. 建立訓練/驗證資料集
# ----------------------------
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

# 儲存 class_names
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# ----------------------------
# 3. 標準化圖片
# ----------------------------
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ----------------------------
# 4. 建立 CNN 模型
# ----------------------------
model = keras.Sequential([
    layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=img_size + (3,)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ----------------------------
# 5. 訓練模型
# ----------------------------
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# ----------------------------
# 6. 測試單張圖片
# ----------------------------
test_img_path = "flower/roses/xxx.jpg"  # 改成你的測試圖片
img = keras.utils.load_img(test_img_path, target_size=img_size)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # 變成 (1, 224, 224, 3)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"This image most likely belongs to {class_names[tf.argmax(score)]} "
      f"with {100 * tf.reduce_max(score):.2f}% confidence.")
