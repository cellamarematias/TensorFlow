# Evaluación de Modelos en Redes Neuronales

Al desarrollar redes neuronales, es esencial seguir un ciclo iterativo:

1. **Construir el modelo**.
2. **Entrenar el modelo**.
3. **Evaluar el modelo**.
4. **Ajustar el modelo**.
5. **Repetir**.

Durante la evaluación, es fundamental recordar: **"Visualizar, visualizar, visualizar"**. Esto implica:

- **Los datos**: ¿Con qué datos estamos trabajando? ¿Cómo se distribuyen?
- **El modelo**: ¿Cuál es la arquitectura del modelo?
- **El entrenamiento**: ¿Cómo evoluciona el rendimiento del modelo durante el aprendizaje?
- **Las predicciones**: ¿Cómo se comparan las predicciones del modelo con los valores reales?

## Creación de un Dataset

Generemos un conjunto de datos simple donde `y = X + 10`:

```
import tensorflow as tf
import matplotlib.pyplot as plt

# Crear datos
X = tf.range(-100, 100, 4)
y = X + 10

# Visualizar datos
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Relación entre X e y')
plt.show()
```

## División del Dataset

Dividimos los datos en conjuntos de entrenamiento y prueba:

- **Conjunto de entrenamiento**: 80% de los datos.
- **Conjunto de prueba**: 20% de los datos.

```
# División de datos
X_train = X[:40]  # 80% para entrenamiento
y_train = y[:40]
X_test = X[40:]   # 20% para prueba
y_test = y[40:]

# Verificar tamaños
len(X_train), len(X_test)
```

## Visualización de la División

Es útil visualizar cómo se han dividido los datos:

```
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c="b", label="Datos de Entrenamiento")
plt.scatter(X_test, y_test, c="g", label="Datos de Prueba")
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('División de Datos en Entrenamiento y Prueba')
plt.show()
```

## Construcción y Compilación del Modelo

Creamos una red neuronal simple con TensorFlow/Keras:

```
# Semilla aleatoria para reproducibilidad
tf.random.set_seed(42)

# Construcción del modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="capa_entrada"),
    tf.keras.layers.Dense(1, name="capa_salida")
], name="modelo_simple")

# Compilación del modelo
modelo.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["mae"]
)

# Resumen del modelo
modelo.summary()
```

**Nota**: Es crucial especificar el parámetro `input_shape` en la primera capa para que el modelo conozca la forma de los datos de entrada y se construya correctamente. De lo contrario, al llamar a `modelo.summary()`, podría generarse un error indicando que el modelo aún no ha sido construido. :contentReference[oaicite:0]{index=0}

## Entrenamiento del Modelo

Entrenamos el modelo con los datos de entrenamiento:

```
# Entrenamiento del modelo
historial = modelo.fit(X_train, y_train, epochs=100, verbose=0)
```

## Evaluación del Modelo

Evaluamos el rendimiento del modelo utilizando los datos de prueba:

```
# Evaluación del modelo
pérdida = modelo.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {pérdida}")
```

## Visualización del Proceso de Entrenamiento

Podemos visualizar cómo evolucionó la pérdida durante el entrenamiento:

```
plt.plot(historial.history['loss'])
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Evolución de la Pérdida durante el Entrenamiento')
plt.show()
```

## Predicciones del Modelo

Finalmente, realizamos predicciones y las comparamos con los valores reales:

```
# Realizar predicciones
y_pred = modelo.predict(X_test)

# Visualizar predicciones vs valores reales
plt.scatter(X_test, y_test, c="g", label="Valores Reales")
plt.scatter(X_test, y_pred, c="r", label="Predicciones")
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparación entre Predicciones y Valores Reales')
plt.show()
```

Este flujo de trabajo proporciona una comprensión clara de cómo construir, entrenar, evaluar y ajustar modelos de redes neuronales, enfatizando la importancia de la visualización en cada paso del proceso.
