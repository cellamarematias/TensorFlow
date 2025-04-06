# ⚙️ ¿Qué es un Optimizer (Optimizador) en una red neuronal?

Un **optimizador** es el algoritmo que se encarga de **ajustar los pesos internos** del modelo para **minimizar la pérdida (loss)** durante el entrenamiento.

# 🧠 ¿Cómo funciona?
Después de cada pasada por los datos (época), el optimizador **modifica los pesos** del modelo basándose en el error que obtuvo. El objetivo es que la **pérdida sea cada vez menor**.

# 📦 Optimizers comunes en TensorFlow/Keras

## ✅ `SGD` - Stochastic Gradient Descent
- El más simple.
- Actualiza los pesos con cada muestra o mini-lote.
- Puede ser lento y sensible a la escala de los datos.

## 🚀 `Adam` - Adaptive Moment Estimation
- Uno de los más usados.
- Combina lo mejor de `SGD` + técnicas como momento y tasa de aprendizaje adaptativa.
- Generalmente **requiere menos ajuste manual**.
- Muy eficiente para muchos tipos de problemas.

## 🌀 Otros:
- `RMSprop`: útil en redes recurrentes (RNN).
- `Adagrad`, `Adadelta`: se adaptan según el parámetro.

# 🧪 ¿Cómo lo usás?
Ejemplo en código:

```
model.compile(
  loss='mae',
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
  metrics=['mae']
)
```

# 📌 En resumen:
- El optimizador **actualiza los pesos** del modelo para que **aprenda mejor**.
- Elegir el correcto puede **mejorar el entrenamiento**.
- `Adam` es una **gran elección por defecto** para comenzar.
