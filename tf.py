import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 關閉 GPU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# 1. 資料夾設定
# ----------------------------
data_dir = "flower"   # 本地花朵資料夾
img_size = (224, 224)
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

# 存成 JSON，Android 端對照使用
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

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
# 6. 儲存 Keras 模型（可選）
# ----------------------------
model_save_path = "flower_cnn_model"
model.save(model_save_path)
print(f"Keras model saved to: {model_save_path}")

# ----------------------------
# 7. 轉換成 TFLite
# ----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 可選量化，加速推論並減小檔案
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_path = "flower_cnn_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {tflite_model_path}")

# ----------------------------
# 8. 測試單張圖片推論
# ----------------------------
test_img_path = "flowers/rose/12240303_80d87f77a3_n.jpg"  # 改成你的測試圖片
img = keras.utils.load_img(test_img_path, target_size=img_size)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # (1, 224, 224, 3)
img_array = img_array / 255.0  # 標準化

# 使用原 Keras 模型做測試
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(f"This image most likely belongs to {class_names[tf.argmax(score)]} "
      f"with {100 * tf.reduce_max(score):.2f}% confidence.")
