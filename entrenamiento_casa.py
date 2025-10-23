import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import os
import joblib

# Configuración
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

print("=== ENTRENAMIENTO CON 80 DATOS - PRECIOS DE CASAS ===")

# =============================================================================
# GENERACIÓN DE 80 DATOS REALISTAS
# =============================================================================

def generar_datos_80():
    """Genera 80 datos realistas de precios de casas"""
    np.random.seed(42)
    
    # 80 tamaños distribuidos de forma realista
    tamanos = np.concatenate([
        np.random.uniform(30, 80, 20),    # Estudios y aptos pequeños
        np.random.uniform(80, 120, 25),   # Aptos familiares
        np.random.uniform(120, 180, 20),  # Casas pequeñas/medianas
        np.random.uniform(180, 300, 10),  # Casas grandes
        np.random.uniform(300, 450, 5)    # Casas lujosas
    ])
    
    # Modelo de precio realista con componente no lineal
    precios = (
        40 +  # Base
        tamanos * 0.9 +  # Componente lineal
        np.power(tamanos, 0.75) * 4 +  # Componente no lineal
        np.random.normal(0, 15, len(tamanos))  # Ruido
    )
    
    # Asegurar precios mínimos realistas
    precios = np.maximum(precios, 50)
    
    return tamanos, precios

print("Generando 80 datos realistas...")
X_full, y_full = generar_datos_80()

# Ordenar por tamaño para mejor visualización
sorted_idx = np.argsort(X_full)
X_full = X_full[sorted_idx]
y_full = y_full[sorted_idx]

print(f"✓ Dataset generado: {len(X_full)} puntos")
print(f"📊 Estadísticas de tamaños:")
print(f"   - Mínimo: {X_full.min():.1f} m²")
print(f"   - Máximo: {X_full.max():.1f} m²") 
print(f"   - Promedio: {X_full.mean():.1f} m²")
print(f"   - Mediana: {np.median(X_full):.1f} m²")

print(f"\n💰 Estadísticas de precios:")
print(f"   - Mínimo: ${y_full.min():.1f} miles")
print(f"   - Máximo: ${y_full.max():.1f} miles")
print(f"   - Promedio: ${y_full.mean():.1f} miles")

# =============================================================================
# ANÁLISIS EXPLORATORIO DE DATOS
# =============================================================================

plt.figure(figsize=(15, 10))

# Gráfico 1: Distribución de datos
plt.subplot(2, 3, 1)
plt.scatter(X_full, y_full, alpha=0.7, color='blue', s=50)
plt.xlabel('Tamaño (m²)')
plt.ylabel('Precio (miles USD)')
plt.title('Distribución de los 80 Datos\nTamaño vs Precio')
plt.grid(True, alpha=0.3)

# Gráfico 2: Histograma de tamaños
plt.subplot(2, 3, 2)
plt.hist(X_full, bins=15, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Tamaño (m²)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Tamaños')
plt.grid(True, alpha=0.3)

# Gráfico 3: Histograma de precios
plt.subplot(2, 3, 3)
plt.hist(y_full, bins=15, alpha=0.7, color='red', edgecolor='black')
plt.xlabel('Precio (miles USD)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Precios')
plt.grid(True, alpha=0.3)

# Gráfico 4: Precio por m²
plt.subplot(2, 3, 4)
precio_por_m2 = y_full / X_full
plt.scatter(X_full, precio_por_m2, alpha=0.7, color='purple', s=50)
plt.xlabel('Tamaño (m²)')
plt.ylabel('Precio por m² (miles USD)')
plt.title('Precio por Metro Cuadrado')
plt.grid(True, alpha=0.3)

# Gráfico 5: Boxplot de tamaños
plt.subplot(2, 3, 5)
plt.boxplot(X_full, vert=True)
plt.ylabel('Tamaño (m²)')
plt.title('Boxplot - Tamaños')
plt.grid(True, alpha=0.3)

# Gráfico 6: Boxplot de precios
plt.subplot(2, 3, 6)
plt.boxplot(y_full, vert=True)
plt.ylabel('Precio (miles USD)')
plt.title('Boxplot - Precios')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_exploratorio_80_datos.png', dpi=300, bbox_inches='tight')
print("✓ Análisis exploratorio guardado: 'analisis_exploratorio_80_datos.png'")

# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

# Dividir en train/validation/test (70%/15%/15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\n📊 DIVISIÓN DE DATOS (80 puntos totales):")
print(f"   • Entrenamiento: {len(X_train)} puntos (70%)")
print(f"   • Validación:    {len(X_val)} puntos (15%)") 
print(f"   • Test:          {len(X_test)} puntos (15%)")

# Escalar los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1))
X_val_scaled = scaler_X.transform(X_val.reshape(-1, 1))
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1))

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Guardar scalers para uso futuro
joblib.dump(scaler_X, 'scaler_X_80.pkl')
joblib.dump(scaler_y, 'scaler_y_80.pkl')

# =============================================================================
# MODELOS NEURONALES
# =============================================================================

def crear_modelo_lineal():
    """Modelo lineal simple"""
    model = keras.Sequential([
        layers.Dense(1, input_shape=(1,), name='salida_lineal')
    ])
    return model

def crear_modelo_red_media():
    """Red neuronal con capacidad media"""
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(1,)),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model

def crear_modelo_red_compleja():
    """Red neuronal más compleja"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model

# Crear modelos
modelos = {
    'Lineal': crear_modelo_lineal(),
    'Red Neuronal Media': crear_modelo_red_media(),
    'Red Neuronal Compleja': crear_modelo_red_compleja()
}

# Callbacks para entrenamiento
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=25, restore_best_weights=True, verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1
)

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

print("\n" + "="*50)
print("INICIANDO ENTRENAMIENTO")
print("="*50)

historiales = {}
predicciones_escaladas = {}

for nombre, modelo in modelos.items():
    print(f"\n🚀 Entrenando: {nombre}")
    
    # Compilar modelo
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )
    
    # Mostrar resumen
    print(f"   Arquitectura: {len(modelo.layers)} capas, {modelo.count_params():,} parámetros")
    
    # Entrenar
    historial = modelo.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=500,
        batch_size=8,  # Batch pequeño para 80 datos
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    historiales[nombre] = historial
    predicciones_escaladas[nombre] = modelo.predict(X_test_scaled, verbose=0)
    
    epocas_entrenadas = len(historial.history['loss'])
    loss_final = historial.history['loss'][-1]
    val_loss_final = historial.history['val_loss'][-1]
    
    print(f"   ✅ Entrenado: {epocas_entrenadas} épocas")
    print(f"   📉 Loss final: {loss_final:.4f} (train), {val_loss_final:.4f} (val)")

# =============================================================================
# EVALUACIÓN
# =============================================================================

print("\n" + "="*50)
print("EVALUACIÓN EN DATOS DE TEST")
print("="*50)

# Convertir predicciones a escala original
predicciones_original = {}
for nombre, pred in predicciones_escaladas.items():
    predicciones_original[nombre] = scaler_y.inverse_transform(pred).flatten()

y_test_original = scaler_y.inverse_transform(y_test_scaled).flatten()

resultados = []

for nombre in modelos.keys():
    y_pred = predicciones_original[nombre]
    
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
    
    resultados.append({
        'Modelo': nombre,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    })
    
    print(f"\n📊 {nombre}:")
    print(f"   • R²: {r2:.4f}")
    print(f"   • RMSE: ${rmse:.2f} miles")
    print(f"   • MAE: ${mae:.2f} miles") 
    print(f"   • Error porcentual: {mape:.2f}%")

# Identificar mejor modelo
resultados_df = pd.DataFrame(resultados)
mejor_modelo_nombre = resultados_df.loc[resultados_df['R²'].idxmax(), 'Modelo']
mejor_modelo = modelos[mejor_modelo_nombre]
mejor_resultado = resultados_df[resultados_df['Modelo'] == mejor_modelo_nombre].iloc[0]

print(f"\n⭐ MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   🎯 R²: {mejor_resultado['R²']:.4f}")
print(f"   💰 Error promedio: ±${mejor_resultado['MAE']:.0f} miles")
print(f"   📈 Precisión: {100-mejor_resultado['MAPE (%)']:.1f}%")

# =============================================================================
# VISUALIZACIÓN DE RESULTADOS
# =============================================================================

plt.figure(figsize=(20, 12))

# 1. Comparación de modelos
plt.subplot(3, 4, 1)
plt.scatter(X_full, y_full, alpha=0.2, color='gray', s=30, label='Todos los datos')
plt.scatter(X_test, y_test_original, color='black', s=50, label='Datos test', alpha=0.8)

colores = ['red', 'blue', 'green']
for i, (nombre, color) in enumerate(zip(modelos.keys(), colores)):
    # Curva suave de predicciones
    X_suave = np.linspace(X_full.min(), X_full.max(), 100)
    X_suave_esc = scaler_X.transform(X_suave.reshape(-1, 1))
    y_suave_esc = modelos[nombre].predict(X_suave_esc, verbose=0)
    y_suave = scaler_y.inverse_transform(y_suave_esc).flatten()
    
    plt.plot(X_suave, y_suave, color=color, linewidth=2, label=nombre)

plt.xlabel('Tamaño (m²)')
plt.ylabel('Precio (miles USD)')
plt.title('Comparación de Modelos - Predicciones')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Pérdidas de entrenamiento
plt.subplot(3, 4, 2)
for nombre, color in zip(modelos.keys(), colores):
    hist = historiales[nombre]
    plt.plot(hist.history['loss'], color=color, alpha=0.8, label=f'{nombre} - Train')
    plt.plot(hist.history['val_loss'], color=color, linestyle='--', alpha=0.8, label=f'{nombre} - Val')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.title('Pérdida durante Entrenamiento')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 3. Métricas de comparación
plt.subplot(3, 4, 3)
metricas = ['RMSE', 'MAE']
x_pos = np.arange(len(modelos))
ancho = 0.35

rmse_valores = [r['RMSE'] for r in resultados]
mae_valores = [r['MAE'] for r in resultados]

plt.bar(x_pos - ancho/2, rmse_valores, ancho, label='RMSE', alpha=0.8, color='lightcoral')
plt.bar(x_pos + ancho/2, mae_valores, ancho, label='MAE', alpha=0.8, color='lightblue')

plt.xlabel('Modelos')
plt.ylabel('Error (miles USD)')
plt.title('Comparación de Errores')
plt.xticks(x_pos, [nombre.replace(' ', '\n') for nombre in modelos.keys()])
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Coeficiente R²
plt.subplot(3, 4, 4)
r2_valores = [r['R²'] for r in resultados]
barras = plt.bar(x_pos, r2_valores, color=colores, alpha=0.8)
plt.xlabel('Modelos')
plt.ylabel('R²')
plt.title('Coeficiente de Determinación R²')
plt.xticks(x_pos, [nombre.replace(' ', '\n') for nombre in modelos.keys()])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

for barra, valor in zip(barras, r2_valores):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.02, 
             f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. Predicciones vs Valores reales
plt.subplot(3, 4, 5)
for nombre, color in zip(modelos.keys(), colores):
    y_pred = predicciones_original[nombre]
    plt.scatter(y_test_original, y_pred, color=color, alpha=0.7, label=nombre, s=50)

min_val = min(y_test_original.min(), min([p.min() for p in predicciones_original.values()]))
max_val = max(y_test_original.max(), max([p.max() for p in predicciones_original.values()]))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfecto')

plt.xlabel('Valor Real (miles USD)')
plt.ylabel('Predicción (miles USD)')
plt.title('Predicciones vs Valores Reales')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Errores de predicción
plt.subplot(3, 4, 6)
for nombre, color in zip(modelos.keys(), colores):
    errores = predicciones_original[nombre] - y_test_original
    plt.scatter(X_test, errores, color=color, alpha=0.7, label=nombre, s=50)
plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
plt.xlabel('Tamaño (m²)')
plt.ylabel('Error (miles USD)')
plt.title('Errores de Predicción')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Mejor modelo - predicción detallada
plt.subplot(3, 4, 7)
X_suave = np.linspace(X_full.min(), X_full.max(), 200)
X_suave_esc = scaler_X.transform(X_suave.reshape(-1, 1))
y_suave_esc = mejor_modelo.predict(X_suave_esc, verbose=0)
y_suave = scaler_y.inverse_transform(y_suave_esc).flatten()

plt.plot(X_suave, y_suave, color='red', linewidth=3, label=f'Mejor: {mejor_modelo_nombre}')
plt.scatter(X_full, y_full, alpha=0.3, color='blue', s=30, label='Datos entrenamiento')
plt.scatter(X_test, y_test_original, color='black', s=50, alpha=0.8, label='Datos test')

plt.xlabel('Tamaño (m²)')
plt.ylabel('Precio (miles USD)')
plt.title(f'Mejor Modelo: {mejor_modelo_nombre}')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Distribución de errores
plt.subplot(3, 4, 8)
datos_errores = []
etiquetas = []
for nombre, color in zip(modelos.keys(), colores):
    errores = predicciones_original[nombre] - y_test_original
    datos_errores.append(errores)
    etiquetas.append(nombre)

plt.boxplot(datos_errores, labels=etiquetas, patch_artist=True,
           boxprops=dict(facecolor='lightgreen', color='black'),
           medianprops=dict(color='red', linewidth=2))
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.ylabel('Error (miles USD)')
plt.title('Distribución de Errores')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 9. Resumen de métricas
plt.subplot(3, 4, 9)
plt.axis('off')
texto_resumen = (
    f"RESUMEN - 80 DATOS\n\n"
    f"MEJOR MODELO: {mejor_modelo_nombre}\n"
    f"• R²: {mejor_resultado['R²']:.4f}\n"
    f"• RMSE: ${mejor_resultado['RMSE']:.0f} miles\n"
    f"• MAE: ${mejor_resultado['MAE']:.0f} miles\n"
    f"• MAPE: {mejor_resultado['MAPE (%)']:.1f}%\n\n"
    f"DATOS:\n"
    f"• Total: 80 puntos\n"
    f"• Train: {len(X_train)} puntos\n"
    f"• Val: {len(X_val)} puntos\n"
    f"• Test: {len(X_test)} puntos\n\n"
    f"RANGOS:\n"
    f"• Tamaños: {X_full.min():.0f}-{X_full.max():.0f} m²\n"
    f"• Precios: ${y_full.min():.0f}-${y_full.max():.0f}K"
)

plt.text(0.1, 0.9, texto_resumen, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# 10. Predicciones para tamaños comunes
plt.subplot(3, 4, 10)
tamanos_comunes = np.array([50, 75, 100, 120, 150, 180, 200, 250, 300])
tamanos_comunes_esc = scaler_X.transform(tamanos_comunes.reshape(-1, 1))
predicciones_comunes_esc = mejor_modelo.predict(tamanos_comunes_esc, verbose=0)
predicciones_comunes = scaler_y.inverse_transform(predicciones_comunes_esc).flatten()

plt.bar(range(len(tamanos_comunes)), predicciones_comunes, color='orange', alpha=0.7)
plt.xlabel('Tamaños Comunes')
plt.ylabel('Precio Predicho (miles USD)')
plt.title('Predicciones para Tamaños Comunes')
plt.xticks(range(len(tamanos_comunes)), [f'{t}m²' for t in tamanos_comunes], rotation=45)
plt.grid(True, alpha=0.3)

# Añadir valores en las barras
for i, (tam, precio) in enumerate(zip(tamanos_comunes, predicciones_comunes)):
    plt.text(i, precio + 10, f'${precio:.0f}K', ha='center', va='bottom', fontsize=8)

# 11. Evolución del learning rate
plt.subplot(3, 4, 11)
for nombre, color in zip(modelos.keys(), colores):
    hist = historiales[nombre]
    if 'lr' in hist.history:
        plt.plot(hist.history['lr'], color=color, label=nombre)
plt.xlabel('Época')
plt.ylabel('Learning Rate')
plt.title('Evolución del Learning Rate')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 12. Métricas de entrenamiento
plt.subplot(3, 4, 12)
plt.axis('off')
texto_metricas = "MÉTRICAS DETALLADAS:\n\n"
for resultado in resultados:
    texto_metricas += f"{resultado['Modelo']}:\n"
    texto_metricas += f"  R²: {resultado['R²']:.4f}\n"
    texto_metricas += f"  RMSE: ${resultado['RMSE']:.1f}K\n"
    texto_metricas += f"  MAE: ${resultado['MAE']:.1f}K\n"
    texto_metricas += f"  MAPE: {resultado['MAPE (%)']:.1f}%\n\n"

plt.text(0.1, 0.9, texto_metricas, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('resultados_completos_80_datos.png', dpi=300, bbox_inches='tight')
print("✓ Resultados completos guardados: 'resultados_completos_80_datos.png'")

# =============================================================================
# GUARDAR MEJOR MODELO Y RESULTADOS
# =============================================================================

# Guardar mejor modelo
mejor_modelo.save('mejor_modelo_80_datos.h5')

# Guardar resultados
resultados_df.to_csv('resultados_modelos_80_datos.csv', index=False)

print(f"\n💾 ARCHIVOS GUARDADOS:")
print(f"   • mejor_modelo_80_datos.h5 (Modelo entrenado)")
print(f"   • scaler_X_80.pkl (Scaler características)")
print(f"   • scaler_y_80.pkl (Scaler target)") 
print(f"   • resultados_modelos_80_datos.csv (Métricas)")
print(f"   • analisis_exploratorio_80_datos.png (Análisis datos)")
print(f"   • resultados_completos_80_datos.png (Resultados)")

# =============================================================================
# PREDICCIONES DE EJEMPLO
# =============================================================================

print("\n" + "="*50)
print("PREDICCIONES DE EJEMPLO CON MEJOR MODELO")
print("="*50)

def predecir_con_modelo(tamano_m2):
    """Función para predecir precios"""
    X_input = np.array([[tamano_m2]])
    X_esc = scaler_X.transform(X_input)
    y_pred_esc = mejor_modelo.predict(X_esc, verbose=0)
    return scaler_y.inverse_transform(y_pred_esc)[0][0]

# Ejemplos de predicción
ejemplos = [55, 85, 110, 140, 175, 210, 280, 350]

print(f"\n🎯 PREDICCIONES ({mejor_modelo_nombre}):")
print("-" * 50)
print(f"{'TAMAÑO':<8} {'PRECIO':<12} {'PRECIO/m²':<12} {'TIPO':<20}")
print("-" * 50)

for tam in ejemplos:
    precio = predecir_con_modelo(tam)
    precio_m2 = precio / tam
    
    if tam < 70: tipo = "Estudio"
    elif tam < 100: tipo = "Apto pequeño"
    elif tam < 140: tipo = "Apto familiar" 
    elif tam < 180: tipo = "Casa pequeña"
    elif tam < 250: tipo = "Casa mediana"
    elif tam < 350: tipo = "Casa grande"
    else: tipo = "Mansión"
    
    print(f"{tam:<8} ${precio:<11.0f} ${precio_m2:<11.2f} {tipo:<20}")

print("\n" + "="*50)
print("¡ENTRENAMIENTO COMPLETADO! 🎉")
print("="*50)
print(f"✅ Modelo entrenado con 80 datos exitosamente")
print(f"✅ Mejor modelo: {mejor_modelo_nombre} (R²: {mejor_resultado['R²']:.4f})")
print(f"✅ Error promedio: ±${mejor_resultado['MAE']:.0f} miles")
print(f"✅ Precisión: {100-mejor_resultado['MAPE (%)']:.1f}%")