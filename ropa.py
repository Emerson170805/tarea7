import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import joblib

# Configuración
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("=== SISTEMA COMPLETO: CLASIFICACIÓN DE ROPA (800x800) + REGRESIÓN LINEAL ===")

# =============================================================================
# PARTE 1: CLASIFICACIÓN DE ROPA CON FASHION MNIST
# =============================================================================

print("\n" + "="*50)
print("PARTE 1: CLASIFICACIÓN DE ROPA - RESOLUCIÓN 800x800")
print("="*50)

# 1️⃣ Cargar dataset Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"✓ Dataset cargado: {X_train.shape}")

# 2️⃣ Redimensionar a 800x800
print("📏 Redimensionando imágenes a 800x800 (esto tomará unos minutos)...")
X_train = np.array([tf.image.resize(img[..., np.newaxis], (800, 800)).numpy() for img in X_train])
X_test = np.array([tf.image.resize(img[..., np.newaxis], (800, 800)).numpy() for img in X_test])

# Normalizar (0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"✓ Nuevas dimensiones: {X_train.shape}, {X_test.shape}")

# 3️⃣ Crear modelo CNN profundo para imágenes grandes
print("\n🧠 Creando arquitectura de red neuronal (para 800x800)...")

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(800,800,1)),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(256, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

print(model.summary())

# 4️⃣ Compilar modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5️⃣ Entrenar modelo
print("\n🚀 Entrenando modelo (puede tardar varios minutos con 800x800)...")
history = model.fit(
    X_train, y_train,
    epochs=5,  # puedes aumentar si tu PC lo soporta
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# 6️⃣ Evaluar
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Precisión final: {test_acc*100:.2f}%")

# 7️⃣ Guardar modelo
model.save("modelo_clasificacion_ropa_800x800.h5")
print("💾 Modelo guardado como 'modelo_clasificacion_ropa_800x800.h5'")

# 8️⃣ Gráficas
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.legend(); plt.title('Precisión')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.legend(); plt.title('Pérdida')

plt.tight_layout()
plt.savefig("resultados_entrenamiento_800x800.png", dpi=300)
print("📊 Resultados guardados en 'resultados_entrenamiento_800x800.png'")

# =============================================================================
# PARTE 2: REGRESIÓN LINEAL
# =============================================================================

print("\n" + "="*50)
print("PARTE 2: REGRESIÓN LINEAL SIMPLE")
print("="*50)

temperatura = np.array([15, 16, 18, 20, 21, 23, 25, 27, 30, 32])
ventas = np.array([500, 520, 560, 580, 600, 640, 680, 700, 760, 800])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    temperatura.reshape(-1, 1), ventas, test_size=0.2, random_state=42
)

model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

r_sq = model_reg.score(X_test_reg, y_test_reg)
mse = mean_squared_error(ventas, model_reg.predict(temperatura.reshape(-1, 1)))

print(f"✓ R²: {r_sq:.4f}  |  MSE: {mse:.2f}")

# Guardar modelo de regresión
joblib.dump(model_reg, "modelo_regresion_ventas.pkl")
print("💾 Modelo de regresión guardado como 'modelo_regresion_ventas.pkl'")

# Gráfico
plt.figure(figsize=(8,5))
plt.scatter(temperatura, ventas, color='blue')
plt.plot(temperatura, model_reg.predict(temperatura.reshape(-1, 1)), color='red')
plt.xlabel("Temperatura (°C)")
plt.ylabel("Ventas (unidades)")
plt.title("Regresión Lineal: Ventas vs Temperatura")
plt.savefig("regresion_lineal_800x800.png", dpi=300)
print("📊 Gráfico de regresión guardado en 'regresion_lineal_800x800.png'")

print("\n🎉 Entrenamiento completado exitosamente")
