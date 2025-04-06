# 📊 Conjunto de Datos: Diabetes
Utilizaremos el conjunto de datos de diabetes, que contiene varias características médicas de pacientes y una medida numérica de la progresión de la enfermedad un año después. Este conjunto de datos es ideal para demostrar cómo una red neuronal puede manejar múltiples características para realizar predicciones.

# 🔄 Flujo del Proceso
Carga y Exploración de Datos: Importaremos el conjunto de datos y exploraremos sus características.​

Preprocesamiento: Normalizaremos los datos para mejorar el rendimiento del modelo.​

División de Datos: Separaremos los datos en conjuntos de entrenamiento y prueba.​

Construcción del Modelo: Crearemos una red neuronal con capas ocultas y el optimizador Adam.​

Entrenamiento: Entrenaremos el modelo y registraremos el historial de pérdida.​

Evaluación y Visualización: Evaluaremos el modelo y visualizaremos los resultados.​

# 🛠️ Implementación en Código

# Red Neuronal para Predicción de Progresión de Diabetes

A continuación, implementaremos una red neuronal utilizando el conjunto de datos de diabetes. Este ejemplo incluye múltiples características por muestra y utiliza el optimizador Adam con varias capas ocultas.

## 1. Importar Librerías Necesarias

```
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

## 2. Cargar y Explorar el Conjunto de Datos

```
# Cargar el conjunto de datos de diabetes
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Mostrar información básica sobre los datos
print(f"Número de muestras: {X.shape[0]}")
print(f"Número de características: {X.shape[1]}")
print(f"Características: {diabetes.feature_names}")
```

## 3. Preprocesamiento de Datos

```
# Normalizar las características para que tengan media 0 y desviación estándar 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 4. Dividir los Datos en Conjuntos de Entrenamiento y Prueba

```
# Dividir los datos en 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

## 5. Construir el Modelo de Red Neuronal

```
# Definir el modelo secuencial
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta 1
    tf.keras.layers.Dense(32, activation='relu'),                                   # Capa oculta 2
    tf.keras.layers.Dense(1)                                                       # Capa de salida
])

# Compilar el modelo con el optimizador Adam y la función de pérdida MAE
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mae',
              metrics=['mae'])
```

## 6. Entrenar el Modelo

```
# Entrenar el modelo con 100 épocas y un tamaño de lote de 16
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)
```

## 7. Evaluar el Modelo

```
# Evaluar el modelo en el conjunto de prueba
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Error Absoluto Medio en el conjunto de prueba: {test_mae:.2f}")
```

## 8. Visualizar la Pérdida Durante el Entrenamiento

```
# Graficar la pérdida durante el entrenamiento y la validación
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MAE)')
plt.legend()
plt.grid(True)
plt.show()
```

## 9. Realizar una Predicción

```
# Seleccionar una muestra del conjunto de prueba para predecir
nueva_muestra = X_test[0].reshape(1, -1)
prediccion = model.predict(nueva_muestra)

print(f"Predicción para la nueva muestra: {prediccion[0][0]:.2f}")
print(f"Valor real: {y_test[0]:.2f}")
```

---

Este ejemplo práctico te permite ver cómo una red neuronal maneja múltiples características para realizar predicciones. Además, la visualización de la pérdida durante el entrenamiento te ayuda a comprender el proceso de aprendizaje del modelo.
