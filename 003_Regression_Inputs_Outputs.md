import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Crear datos con 3 características (features) por muestra
# 15 muestras, 3 features cada una
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

# 2️⃣ Crear etiquetas (label): por ejemplo, la suma de las 3 features + 5
y = np.sum(X, axis=1, keepdims=True) + 5  # Resultado será un vector columna (15, 1)

# 3️⃣ Crear modelo con múltiples capas ocultas
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),  # Capa oculta 1
    tf.keras.layers.Dense(8, activation='relu'),   # Capa oculta 2
    tf.keras.layers.Dense(1)                       # Capa de salida
])

# 4️⃣ Compilar el modelo
model.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mae'])

# 5️⃣ Entrenar el modelo
history = model.fit(X, y, epochs=200, verbose=0)

# 6️⃣ Hacer una predicción con una nueva muestra
nueva_muestra = np.array([[20.0, 21.0, 22.0]], dtype=np.float32)
prediccion = model.predict(nueva_muestra)
print(f"📈 Predicción para {nueva_muestra}: {prediccion[0][0]:.2f}")

# 7️⃣ Graficar la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida')
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()
