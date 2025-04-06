# 📏 ¿Cómo se realiza la normalización de los datos?

La **normalización** es un proceso para escalar los datos de entrada (features) para que tengan una media cercana a 0 y una desviación estándar cercana a 1. Esto es muy útil para mejorar el rendimiento y la estabilidad de los modelos de machine learning.

---

## 🧮 Fórmula común: Z-score (Estandarización)

Cada valor ( x ) se transforma así:

```
x_normalizado = (x - media) / desviación_estándar
```

---

## ✅ ¿Por qué normalizamos?

- Algunos algoritmos (como redes neuronales) son sensibles a la escala de los datos.
- Evita que features con valores grandes dominen sobre otras.
- Mejora la convergencia del entrenamiento.

---

## 🧪 Ejemplo práctico con `sklearn`:

```
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Cargar dataset
diabetes = load_diabetes()
X = diabetes.data  # features (ya normalizadas en este caso)
y = diabetes.target  # labels (valores a predecir)

# Si no estuvieran normalizadas, las normalizamos así:
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

---

## 📊 Visualizar el cambio

```
import matplotlib.pyplot as plt
plt.hist(X[:, 0], bins=30, alpha=0.5, label='Original')
plt.hist(X_normalized[:, 0], bins=30, alpha=0.5, label='Normalizado')
plt.legend()
plt.title("Distribución antes y después de la normalización")
plt.show()
```

---

## 🧠 ¿Qué columnas se normalizan?

En general, **todas las columnas numéricas de entrada (features)**. No se normaliza la variable objetivo (`y`) a menos que el modelo lo requiera específicamente.

---

## 🧬 Notas extra

- En el dataset de diabetes de `sklearn`, las features **ya vienen normalizadas**.
- Si usás tus propios datos, siempre normalizá antes de entrenar.

