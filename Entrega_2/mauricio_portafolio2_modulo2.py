# -*- coding: utf-8 -*-

#Momento de Retroalimentación: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)
#Mauricio Guerrero González A01751436
"""###1. Implementación de un Árbol de Decisión con scikit-learn

El objetivo del presente código consiste en predecir si un pasajero sobrevivió al hundimiento del Titanic utilizando un árbol de decisión mediante la biblioteca scikit-learn.

"""

"""###2. Importación de librerías

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""###3. Carga de los datasets

`train.csv` contiene las características de los pasajeros y la variable
objetivo `Survived`.

"""

ruta_train = r"C:\Users\mauri\Downloads\train.csv"
train = pd.read_csv(ruta_train)
print("Tamaño de train.csv:", train.shape)
print("\nTrain")
print(train.head())

"""###4. Identificación de la variable objetivo

La variable objetivo es `Survived`.

Esta variable representa si el pasajero sobrevivió, donde 0 significa que no sobrevivió y 1 representa que sí sobrevivió.

"""

print("\nVariable objetivo: Survived")
print(train["Survived"].value_counts())
print("\nProporción:")
print(train["Survived"].value_counts(normalize=True))

"""###5. Selección de variables

Selecciona variables numéricas y categóricas que pueden aportar información para predecir la supervivencia.

"""

columnas_numericas = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
columnas_categoricas = ["Sex", "Embarked"]
columnas = columnas_numericas + columnas_categoricas
X = train[columnas]
y = train["Survived"]
print("\nVariables utilizadas:")
print(columnas)

"""###6. Separación de entrenamiento y prueba

Separa el dataset en 80 % para entrenamiento y 20 % para prueba.

"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("\nEntrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)

"""###7. Preparación de los datos

Prepara los valores faltantes y transforma las variables categóricas para que puedan ser utilizadas por el árbol de decisión.

"""

preprocesamiento = ColumnTransformer(transformers=[("numericas", SimpleImputer(strategy="median"), columnas_numericas), ("categoricas", Pipeline(steps=[("imputacion", SimpleImputer(strategy="most_frequent")), ("codificacion", OneHotEncoder(handle_unknown="ignore"))]), columnas_categoricas)])

"""###8. Configuración del árbol de decisión

Configura el árbol utilizando el criterio Gini y limita su profundidad para reducir el riesgo de sobreajuste.

"""

modelo = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_split=30, min_samples_leaf=15, random_state=42)

"""###9. Integración del modelo

Integra el preprocesamiento y el árbol de decisión en un solo flujo de trabajo.

"""

pipeline = Pipeline(steps=[("preprocesamiento", preprocesamiento), ("arbol", modelo)])

"""###10. Entrenamiento del árbol

Entrena el árbol de decisión utilizando el conjunto de entrenamiento.

"""

pipeline.fit(X_train, y_train)

"""###11. Generación de predicciones

Genera predicciones para los conjuntos de entrenamiento y prueba.

"""

pred_train = pipeline.predict(X_train)
pred_test = pipeline.predict(X_test)
print("\nPrimeras 20 predicciones:")
print(pred_test[:20])
print("\nPrimeras 20 clases reales:")
print(y_test.iloc[:20].to_numpy())

"""###12. Cálculo de métricas

Calcula Accuracy, Precision, Recall y F1-score para evaluar el desempeño del modelo.

"""

# Calcula las métricas del conjunto de entrenamiento.
accuracy_train = accuracy_score(y_train, pred_train)
precision_train = precision_score(y_train, pred_train, zero_division=0)
recall_train = recall_score(y_train, pred_train, zero_division=0)
f1_train = f1_score(y_train, pred_train, zero_division=0)

# Calcula las métricas del conjunto de prueba.
accuracy_test = accuracy_score(y_test, pred_test)
precision_test = precision_score(y_test, pred_test, zero_division=0)
recall_test = recall_score(y_test, pred_test, zero_division=0)
f1_test = f1_score(y_test, pred_test, zero_division=0)

print("\nENTRENAMIENTO")
print(f"Accuracy : {accuracy_train:.4f}")
print(f"Precision: {precision_train:.4f}")
print(f"Recall   : {recall_train:.4f}")
print(f"F1-score : {f1_train:.4f}")
print("\nPRUEBA")
print(f"Accuracy : {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall   : {recall_test:.4f}")
print(f"F1-score : {f1_test:.4f}")

"""###13. Matriz de confusión

Calcula la matriz de confusión para identificar las clasificaciones correctas e incorrectas del conjunto de prueba.

"""

# Calcula la matriz de confusión del conjunto de prueba.
matriz_test = confusion_matrix(y_test, pred_test)
print("\nMatriz de confusión:")
print(matriz_test)

"""###14. Visualización de la matriz de confusión

Muestra la matriz de confusión obtenida por el modelo.

"""

plt.figure(figsize=(7, 6))
plt.imshow(matriz_test)
plt.colorbar()
plt.xticks([0, 1], ["No sobrevive", "Sobrevive"])
plt.yticks([0, 1], ["No sobrevive", "Sobrevive"])
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.title("Matriz de Confusión")

for i in range(2):
    for j in range(2):
        plt.text(j, i, matriz_test[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

"""###15. Visualización del árbol de decisión

Muestra la estructura del árbol generado por el modelo.

"""

# Obtiene el árbol entrenado y los nombres de las variables transformadas.
modelo_entrenado = pipeline.named_steps["arbol"]
nombres_variables = pipeline.named_steps["preprocesamiento"].get_feature_names_out()

plt.figure(figsize=(18, 9))
plot_tree(modelo_entrenado, feature_names=nombres_variables, class_names=["No sobrevive", "Sobrevive"], filled=True, rounded=True, max_depth=4, fontsize=8)
plt.title("Árbol de Decisión")
plt.tight_layout()
plt.show()

"""###16. Predicciones de ejemplo

Muestra algunas observaciones del conjunto de prueba junto con su clase real y la predicción obtenida.

"""

ejemplos = X_test.head(15).copy()
ejemplos["Real"] = y_test.iloc[:15].to_numpy()
ejemplos["Prediccion"] = pred_test[:15]
ejemplos["Correcta"] = ejemplos["Real"] == ejemplos["Prediccion"]
print("\nPredicciones de ejemplo:")
print(ejemplos)