# 🔍 Red Neuronal con múltiples features, Adam, y capas ocultas

A continuación, un ejemplo completo y comentado paso a paso en TensorFlow/Keras.

## 📦 Paso 1: Importar librerías

```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

## 📊 Paso 2: Crear los datos (features y labels)

Supongamos que tenemos 3 features por muestra (x1, x2, x3) y queremos predecir un solo número.

```
# Creamos un dataset simple con 15 muestras y 3 características por muestra
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10],
    [9, 10, 11],
    [10, 11, 12],
    [11, 12, 13],
    [12, 13, 14],
    [13, 14, 15],
    [14, 15, 16],
    [15, 16, 17]
], dtype=np.float32)

# El label es simplemente la suma de las tres características
y = np.sum(X, axis=1)  # Resultado esperado: [6, 9, 12, ..., 48]
```

## 🧠 Paso 3: Definir el modelo

```
model = tf.keras.Sequential([
    # Capa oculta 1 con 64 neuronas y ReLU
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    
    # Capa oculta 2 con 32 neuronas
    tf.keras.layers.Dense(32, activation='relu'),
    
    # Capa de salida con 1 neurona (regresión)
    tf.keras.layers.Dense(1)
])
```

## ⚙️ Paso 4: Compilar el modelo

```
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Usamos Adam con una tasa de aprendizaje
    loss='mae',   # Mean Absolute Error (bueno para regresión simple)
    metrics=['mae']
)
```

## 🔁 Paso 5: Entrenar el modelo

```
history = model.fit(
    X, y, 
    epochs=200,         # Número de pasadas completas por los datos
    batch_size=4,       # Cuántas muestras por actualización de pesos
    verbose=1           # Muestra el progreso durante el entrenamiento
)
```

## 📈 Paso 6: Visualizar la pérdida

```
plt.plot(history.history['loss'])
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Loss (MAE)")
plt.grid(True)
plt.show()
```

## 🔮 Paso 7: Hacer una predicción

```
# Queremos predecir para esta muestra: x1=20, x2=21, x3=22 → suma = 63
nueva_muestra = np.array([[20.0, 21.0, 22.0]], dtype=np.float32)
prediccion = model.predict(nueva_muestra)

print(f"📈 Predicción para {nueva_muestra[0]}: {prediccion[0][0]}")
```

---

Este ejemplo muestra cómo usar **múltiples características por muestra**, cómo estructurar un modelo de regresión, y cómo interpretar cada parte.

