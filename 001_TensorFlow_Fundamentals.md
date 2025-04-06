# Módulo 1: Deep Learning and TensorFlow Fundamentals

## 6. What is deep learning?
**Resumen:**  
Deep learning es una rama del machine learning que utiliza redes neuronales profundas (varias capas) para aprender representaciones complejas a partir de grandes volúmenes de datos.

**Ejemplo:**  
Imagina una red que reconoce imágenes de gatos y perros; la red aprende de las características (bordes, texturas) de cada imagen a través de múltiples capas.

---

## 7. Why use deep learning?
**Resumen:**  
Se utiliza deep learning porque puede modelar relaciones complejas y patrones en datos no estructurados (imágenes, audio, texto), superando en muchos casos a los métodos tradicionales.

**Ejemplo:**  
Aplicar una red profunda para reconocer voz, lo que permite transcribir audio a texto con alta precisión.

---

## 8. What are neural networks?
**Resumen:**  
Las redes neuronales son sistemas computacionales inspirados en el cerebro humano, formados por neuronas (nodos) conectadas en capas que procesan y transforman la información.

**Ejemplo:**  
Un perceptrón simple que recibe varios inputs, los pondera y produce un output basado en una función de activación.

---

## 9. Python + Machine Learning Monthly
**Resumen:**  
Este capítulo resalta la importancia de Python en el ecosistema de machine learning, mostrando cómo bibliotecas como TensorFlow y NumPy simplifican el desarrollo y la experimentación.

**Ejemplo:**  
Utilizar Python para cargar datos, procesarlos con NumPy y construir modelos con TensorFlow.

---

## 10. What is deep learning already being used for?
**Resumen:**  
Se exploran aplicaciones actuales de deep learning, como visión por computadora, procesamiento del lenguaje natural, reconocimiento de voz y sistemas de recomendación.

**Ejemplo:**  
Uso de redes convolucionales (CNN) para clasificar imágenes en aplicaciones médicas o de seguridad.

---

## 11. What is and why use TensorFlow?
**Resumen:**  
Se introduce TensorFlow, una biblioteca de código abierto que permite construir y entrenar modelos de deep learning de manera escalable y eficiente.

**Ejemplo:**  
Construir un modelo simple de clasificación utilizando `tf.keras` para predecir etiquetas de imágenes.

---

## 12. What is a Tensor?
**Resumen:**  
Un tensor es una estructura de datos multidimensional (como matrices y vectores) y es el objeto fundamental en TensorFlow para representar datos.

**Ejemplo:**
```python
import tensorflow as tf
tensor = tf.constant([1, 2, 3])
print(tensor)
```

---

## 13. What we're going to cover throughout the course
**Resumen:**  
Se presenta el temario del curso, abarcando desde operaciones básicas con tensores hasta técnicas avanzadas en deep learning, explicando el flujo de contenidos.

**Ejemplo:**  
Una hoja de ruta que muestra los módulos y capítulos que se irán explorando a lo largo del curso.

---

## 14. How to approach this course
**Resumen:**  
Se ofrecen estrategias y consejos para abordar el curso, enfatizando la importancia de la práctica y la experimentación con código en TensorFlow.

**Ejemplo:**  
Sugerencia de realizar ejercicios prácticos y proyectos pequeños tras cada capítulo para reforzar el aprendizaje.

---

## 15. Need A Refresher?
**Resumen:**  
Un breve repaso de conceptos previos, como fundamentos de Python y matemáticas básicas, necesarios para aprovechar el contenido del curso.

**Ejemplo:**  
Recordar operaciones aritméticas y conceptos básicos de álgebra lineal (vectores y matrices).

---

## 16. Creating your first tensors with TensorFlow and tf.constant()
**Resumen:**  
Aprender a crear tensores inmutables usando `tf.constant()`.

**Ejemplo:**
```python
tensor_const = tf.constant([4, 5, 6])
print(tensor_const)
```

---

## 17. Creating tensors with TensorFlow and tf.Variable()
**Resumen:**  
Se muestra cómo crear tensores mutables con `tf.Variable()`, que permiten modificar su contenido durante el entrenamiento o procesamiento.

**Ejemplo:**
```python
tensor_var = tf.Variable([4, 5, 6])
tensor_var.assign([7, 8, 9])
print(tensor_var)
```

---

## 18. Creating random tensors with TensorFlow
**Resumen:**  
Generar tensores con valores aleatorios, útil para inicializar pesos en redes neuronales.

**Ejemplo:**
```python
random_tensor = tf.random.uniform([2, 3])
print(random_tensor)
```

---

## 19. Shuffling the order of tensors
**Resumen:**  
Aprender a mezclar (shuffle) el orden de los datos en un tensor, un paso común en la preparación de datos para entrenar modelos.

**Ejemplo:**
```python
shuffled_tensor = tf.random.shuffle(tf.constant([1, 2, 3, 4, 5]))
print(shuffled_tensor)
```

---

## 20. Creating tensors from NumPy arrays
**Resumen:**  
Convertir arreglos de NumPy a tensores de TensorFlow, facilitando la interoperabilidad entre ambas bibliotecas.

**Ejemplo:**
```python
import numpy as np
np_array = np.array([[1, 2], [3, 4]])
tensor_from_np = tf.convert_to_tensor(np_array)
print(tensor_from_np)
```

---

## 21. Getting information from your tensors (tensor attributes)
**Resumen:**  
Acceder a atributos importantes de un tensor, como su forma, tipo de dato y número de dimensiones.

**Ejemplo:**
```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print("Shape:", tensor.shape)
print("Dtype:", tensor.dtype)
```

---

## 22. Indexing and expanding tensors
**Resumen:**  
Aprender a acceder a elementos específicos mediante indexación y a modificar la forma del tensor agregando dimensiones cuando sea necesario.

**Ejemplo:**
```python
tensor = tf.constant([1, 2, 3])
print("Primer elemento:", tensor[0])
expanded_tensor = tf.expand_dims(tensor, axis=0)
print("Expanded:", expanded_tensor)
```

---

## 23. Manipulating tensors with basic operations
**Resumen:**  
Realizar operaciones matemáticas básicas (suma, resta, multiplicación, división) directamente sobre tensores.

**Ejemplo:**
```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
result = tf.add(a, b)
print(result)
```

---

## 24. Matrix multiplication with tensors part 1
**Resumen:**  
Introducción a la multiplicación de matrices utilizando tensores, una operación central en deep learning.

**Ejemplo:**
```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
result = tf.matmul(a, b)
print(result)
```

---

## 25. Matrix multiplication with tensors part 2
**Resumen:**  
Se profundiza en la multiplicación de matrices, explorando casos de broadcasting y diferentes configuraciones de formas (shapes) de los tensores.

**Ejemplo:**  
Multiplicación entre tensores de diferentes dimensiones, asegurando la compatibilidad para el cálculo.

---

## 26. Matrix multiplication with tensors part 3
**Resumen:**  
Exploración de técnicas avanzadas para la multiplicación de matrices, incluyendo optimizaciones y transformaciones (como trasposición) para mejorar la eficiencia.

**Ejemplo:**
```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[7, 8], [9, 10], [11, 12]])
result = tf.matmul(a, b)
print(result)
```
