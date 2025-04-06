# 📦 ¿Qué es el Batch Size en el entrenamiento de una red neuronal?

El **batch size** (tamaño del lote) es la **cantidad de muestras** que el modelo procesa **antes de actualizar sus pesos** durante el entrenamiento.

# 🧠 ¿Cómo funciona?
Cuando entrenás un modelo con datos grandes, no se procesan todos a la vez. En cambio, los datos se dividen en pequeños **"batches"**.

Ejemplo:
- Tenés 1000 muestras.
- Batch size = 32.
- Entonces, el modelo procesará **32 muestras a la vez**, y luego ajustará los pesos.

# 🔄 Relación con épocas (epochs)
- Una **época** es una pasada completa por todos los datos.
- Si tenés 1000 muestras y batch size = 100, entonces habrá **10 pasos** (batches) por época.

# ⚖️ ¿Qué batch size elegir?
- **Batch pequeño (e.g. 8, 16):** más actualizaciones, más ruido, pero puede ayudar a salir de mínimos locales.
- **Batch grande (e.g. 64, 128):** más estable pero puede consumir más memoria y generalizar menos.

# 💡 Por defecto en Keras
Si no lo especificás, Keras usará **batch size = 32**.

# 🧪 Ejemplo en código
```
model.fit(X, y, epochs=100, batch_size=16)
```

# 📌 En resumen:
- El batch size **controla cuántas muestras se usan antes de actualizar los pesos**.
- Influye en la **velocidad, memoria y calidad** del entrenamiento.
