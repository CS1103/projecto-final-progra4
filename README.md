[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Implementación de una red neuronal para deducir el resultado de  la Suma de 2 números con 2 dígitos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Uso y creación de redes neuronales.
* **Grupo**: `Progra4`
* **Integrantes**:

  * Candy Zuta Vargas    – 201810725  (Responsable de investigación teórica)
  * Valeria Gadea Lock   – 202010300 (Desarrollo de la arquitectura)
  * Roger Zavaleta Alvino – 202010438 (Implementación del modelo)
  * Valeria Gadea Lock – 202010300 (Pruebas y benchmarking)
  * Roger Zavaleta Alvino – 202010438 (Documentación y demo)



---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:
   * CMake 3.10.2+
3. **Instalación**:

   ```bash
   git clone https://github.com/CS1103/projecto-final-progra4.git
   cd projecto-final-progra4
   mkdir build && cd build
   cmake ..
   make
   ```



---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido**:
  1. Historia y evolución de las NNs.
  * El origen de las redes neuronales artificiales se remonta a la década de 1940, con el modelo de McCulloch y Pitts, que planteó una primera representación matemática simplificada de la neurona biológica. Durante los años 50 y 60, surgieron avances como los perceptrones de Rosenblatt, que demostraron la capacidad de aprender patrones básicos. Sin embargo, limitaciones en su capacidad de resolver problemas no lineales provocaron un período de menor interés. El desarrollo del algoritmo de retropropagación en los años 80 permitió el entrenamiento efectivo de redes multicapa, reactivando la investigación. Desde los 2000, la mejora de la capacidad de cómputo y el auge del deep learning han impulsado la adopción masiva de redes neuronales profundas en numerosos campos, incluyendo procesamiento de imágenes, lenguaje natural y predicciones numéricas.

  2. Principales arquitecturas: MLP, CNN, RNN.
  * Perceptrón Multicapa (MLP): Formado por varias capas de neuronas totalmente conectadas y funciones de activación no lineales. Se utiliza en tareas de clasificación y regresión sobre datos tabulares o vectoriales.
  * Redes Convolucionales (CNN): Incorporan operaciones de convolución y pooling para extraer características espaciales jerárquicas, principalmente en imágenes.
  * Redes Recurrentes (RNN): Añaden conexiones que modelan relaciones temporales en secuencias, siendo adecuadas para procesamiento de texto y series temporales.
En este proyecto se implementa un MLP, dado que es la arquitectura más adecuada para aprender la relación numérica entre los dígitos de entrada y la suma esperada.}

  3. Algoritmos de entrenamiento: backpropagation, optimizadores.
  * El aprendizaje de una red neuronal consiste en ajustar los parámetros internos para reducir el error entre las salidas predichas y las reales. El método más habitual es la retropropagación del error (backpropagation), que calcula los gradientes de la función de pérdida respecto a cada peso. Posteriormente, se aplican algoritmos de optimización como:
    * SGD (Stochastic Gradient Descent): Ajusta cada peso en función del gradiente y de una tasa de aprendizaje fija.
    * Adam: Calcula tasas de aprendizaje adaptativas y momentums por parámetro, logrando una convergencia más rápida y estable en muchos casos.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**
  * Strategy (definir algoritmos intercambiables):
    * Optimizadores (SGD, Adam)
    * Funciones de pérdida (MSELoss, BCELoss)
    * Capas (Dense, ReLU, Sigmoid)
  * Composite  (Facilita la construcción de arquitecturas complejas):
    * Clase NeuralNetwork contiene múltiples objetos ILayer
  * Factory (La creación de objetos se delega a funciones específicas):
    * Uso de std::make_unique para crear instancias de ILayer
  * Iterator  (Uso de iteradores estándar para recorrer estructuras):
    * Uso de iteradores para la clase tensor.h.
  * Decorator:
    * Uso para decorar de forma  parcial la clase  nn_dense.h al ser utilizado por nn_activation.h.
* **Estructura de archivos**:

  ```
  projecto-final-progra4/
  ├── tensor.h
  ├── nn_optimizer.h
  ├── nn_loss.h
  ├── nn_layer.h
  ├── nn_interfaces.h
  ├── nn_dense.h
  ├── nn_activation.h
  ├── neural_network.h
  ├── main.cpp
  ├──video/demo.mp4
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: 

  * Se compila el programa, ya sea utilizando g++ o el IDE de su preferencia .

  * El programa presenta un menú principal que le permite entrenar la red neuronal
    en base a parámetros por defecto.
  * El programa permite realizar el entrenamiento con parámetros personalizados
    y probar la red neuronal de forma interactiva al ingresar las sumas a probar.

  
---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `video/demo.mp4`.
> Pasos:
>
> 1. Explicacion del funcionamiento de la red neuronal.
> 2. Ejecutar comando de entrenamiento en el menú principla.
> 3. Evaluar resultados mediante tests automaticos o manuales al finalizar el entrenamiento  

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo (entrenamiento por defecto**:

  * Iteraciones: 15 épocas.
  * Tiempo total de entrenamiento:  2s289ms
  * Presisión total: 50%.

* **Métricas de ejemplo (entrenamiento personalizado 1**:
  * IParámetros de la red neuronal: por defecto.
  * Iteraciones: 30 épocas.
  * Tiempo total de entrenamiento: 4s529ms.
  * Presisión total: 83.33333%
* **Métricas de ejemplo (entrenamiento personalizado 2**:
  * IParámetros de la red neuronal: por defecto.
  * Iteraciones: 60 épocas.
  * Tiempo total de entrenamiento: 9s091ms.
  * Presisión total: 91.6667%.
* **Ventajas/Desventajas**:

  * * Manejo de errores en entrada del usuario
  * * Personalización avanzada para experimentación
  * * Configuración por defecto para uso sencillo
  * * Suficiente capacidad (neuronas/capas) para aprender patrones complejos
  * – Sin paralelización, rendimiento limitado.
  * – Sin carga de archivos de prueba para pruebas masivas personalizadas
  * – Falta de una implementación para puardar el progreso del modelo entrenado 
  para permitir ser entrenado más de 1 vez para la mejora contrinua de la red neuronal.
* **Mejoras futuras**:

  * Uso de CUDA para acelerar el entrenamiento con acerleradores gráficos
  * Paralelizar el proceso de entrenamiento al aprovechar los demás núcleos 
  del CPU presente en una Pc para dividir el calculo por lotes.


---

### 5. Trabajo en equipo

| Tarea                     | Miembro               | Rol                       |
| ------------------------- |-----------------------| ------------------------- |
| Investigación teórica     | Candy Zuta Vargas     | Documentar bases teóricas |
| Diseño de la arquitectura | Valeria Gadea Lock    | UML y esquemas de clases  |
| Implementación del modelo | Roger Zavaleta Alvino | Código C++ de la NN       |
| Pruebas y benchmarking    | Roger Zavaleta Alvino | Generación de métricas    |
| Documentación y demo      | Todos los integrantes | Tutorial y video demo     |


---

### 6. Conclusiones

* **Logros**:  Demuestrar cómo una red neuronal puede aprender una operación matemática no trivial,
* **Evaluación**: La red neuronal a implementar es capáz de ser utilizada para problemas más complejos.
* **Aprendizajes**: Profundización en backpropagation, optimización mediante Adani y Arquitectura de una red neuronal..
* **Recomendaciones**: Utilizar la red neuronal para problemas mas complejos y optimizar el entrenamiento para un mejor resultado con un menor tiempo.

---

### 7. Bibliografía

> [1] E. B. S. Martín, F. Sáez-Delgado, y N. Lepe-Martínez, "El rol predictivo de la red neuronal por defecto sobre la atención sostenida en edades escolares: una revisión sistemática," Revista chilena de neuro-psiquiatría, vol. 61, no. 1, pp. 87-97, 2023.
> 
> [1] M. Cilimkovic, "Neural networks and back propagation algorithm," Institute of Technology Blanchardstown, vol. 15, no. 1, p. 18, 2015.
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
