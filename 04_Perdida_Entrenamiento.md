# 📉 ¿Qué es la "pérdida" (loss) en el entrenamiento de una red neuronal?
La pérdida (en inglés, loss) es una medida numérica de qué tan mal está prediciendo el modelo durante el entrenamiento.

# 🧠 ¿Qué representa?
Es el error promedio entre los valores predichos por el modelo (y_pred) y los valores reales o verdaderos (y_true).

Por ejemplo, si estamos entrenando un modelo que predice:

Entrada: 10

Etiqueta esperada (y): 20

El modelo predice: 16

Entonces el error es: |20 - 16| = 4 (si usamos MAE, error absoluto medio).

# 🔧 ¿Para qué sirve?
Durante el entrenamiento, el modelo ajusta sus pesos internos para minimizar esta pérdida. Mientras más baja la pérdida, mejor el modelo.

# 📊 ¿Por qué a veces no es 0?
Porque el modelo está aproximando, no memorizando.

Porque hay ruido en los datos.

Porque todavía no se ha entrenado suficiente.

# 📈 ¿Cómo se visualiza?
Podés graficarla así:

```
plt.plot(history.history['loss'])
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Loss (MAE)")
plt.grid(True)
plt.show()
```

# ✅ Tipos comunes de pérdida:
mae: Error Absoluto Medio (Mean Absolute Error)

mse: Error Cuadrático Medio (Mean Squared Error)

binary_crossentropy, categorical_crossentropy: para clasificación

# 📌 En resumen:
La pérdida te dice qué tan bien está aprendiendo tu modelo. Un valor más bajo = mejor rendimiento.