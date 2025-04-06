# 🧱 ¿Qué son las Capas Ocultas (Hidden Layers) en una red neuronal?

Las **capas ocultas** son los **bloques intermedios** entre la capa de entrada (input) y la capa de salida (output) de una red neuronal. Son las responsables de **aprender representaciones complejas** de los datos.

# 🧠 ¿Qué hacen realmente?

Cada capa oculta:
- Toma los datos de la capa anterior.
- Los **transforma aplicando una función matemática** (operación lineal + función de activación).
- Pasa el resultado a la siguiente capa.

Así, capa tras capa, la red puede **aprender relaciones cada vez más abstractas**.

Por ejemplo:
- Primeras capas podrían detectar líneas o bordes (en imágenes).
- Capas más profundas podrían detectar formas completas o patrones complejos.

# ⚙️ ¿Qué contiene una capa oculta?

Una capa oculta tiene:
- **Neuronas (unidades)**: cada una con sus propios **pesos** y **bias**.
- Una **función de activación**, que decide si la neurona "dispara" o no (como una puerta lógica).

La operación matemática básica de cada neurona es:

\[
output = \text{activation}(w_1 \cdot x_1 + w_2 \cdot x_2 + ... + b)
\]

Donde:
- \(x_i\) son las entradas
- \(w_i\) los pesos
- \(b\) el bias
- activation puede ser `relu`, `sigmoid`, `tanh`, etc.

# 🔥 ¿Qué funciones de activación se usan?

Las más comunes:
- `ReLU` (Rectified Linear Unit): muy usada en hidden layers.
- `Sigmoid`: aplasta valores entre 0 y 1 (menos usada en capas ocultas).
- `Tanh`: valores entre -1 y 1.

Ejemplo:

```
tf.keras.layers.Dense(64, activation='relu')
```

# 🧪 ¿Cómo se agregan capas ocultas?

Ejemplo de red con 2 capas ocultas:

```
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1)  # Capa de salida
])
```

# 🎯 ¿Cuántas capas y neuronas usar?

- No hay una respuesta única.
- Se eligen probando (experimentación).
- Muy pocas capas → el modelo puede ser demasiado simple.
- Demasiadas capas → puede sobreajustar (memorizar sin generalizar).

# 📌 En resumen:

- Las **capas ocultas** permiten que la red **aprenda patrones complejos**.
- Cuantas más capas y neuronas, más poder de representación (pero también más riesgo de sobreentrenar).
- Son **esenciales** para que las redes neuronales resuelvan tareas no lineales y profundas.
