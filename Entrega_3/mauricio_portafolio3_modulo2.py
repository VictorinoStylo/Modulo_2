# -*- coding: utf-8 -*-

"""Momento de Retroalimentación: Análisis y Reporte sobre el desempeño del modelo.
Mauricio Guerrero González A01751436

Esta implementación compara el árbol de decisión original contra una versión optimizada. 
"""

"""###1. Importación de librerías"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


"""###2. Carga del dataset

Identifica el archivo train.csv en la ruta especificada. 
"""

ruta_train = r"C:\Users\mauri\Downloads\train.csv"
train = pd.read_csv(ruta_train)
print("Tamaño de train.csv:", train.shape)
print("\nTrain")
print(train.head())


"""###3. Identificación de la variable objetivo

Identifica Survived como la variable objetivo.
0 representa que el pasajero no sobrevivió y 1 representa que sobrevivió.
"""

y = train["Survived"]

print("\nVariable objetivo: Survived")
print(train["Survived"].value_counts())
print("\nProporción:")
print(train["Survived"].value_counts(normalize=True))


"""###4. Identificación de características

Identifica información adicional del dataset que puede ayudar al árbol a representar patrones de supervivencia que no estaban disponibles en la implementación original.

"""

trabajo = train.copy()

# Detecta el título que aparece en el nombre de cada pasajero.
trabajo["Title"] = trabajo["Name"].str.extract(r",\s*([^.]*)\.")[0].str.strip()
trabajo["Title"] = trabajo["Title"].replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)
trabajo["Title"] = trabajo["Title"].where(
    trabajo["Title"].isin(["Mr", "Miss", "Mrs", "Master"]),
    "Rare"
)

# Detecta el número total de integrantes de la familia del pasajero.
trabajo["FamilySize"] = trabajo["SibSp"] + trabajo["Parch"] + 1

# Detecta si el pasajero viajaba sin familiares registrados.
trabajo["IsAlone"] = (trabajo["FamilySize"] == 1).astype(int)

# Identifica una posible madre adulta dentro del conjunto de características disponibles.
trabajo["Mother"] = (
    (trabajo["Sex"] == "female")
    & (trabajo["Parch"] > 0)
    & (trabajo["Age"] > 18)
    & (trabajo["Title"] == "Mrs")
).astype(int)

# Detecta la tarifa aproximada pagada por cada integrante de la familia.
trabajo["FarePerPerson"] = trabajo["Fare"] / trabajo["FamilySize"]

# Detecta la cubierta del barco a partir de la primera letra del número de cabina.
trabajo["Deck"] = trabajo["Cabin"].fillna("U").str[0]


"""###5. Selección de variables

Identifica las variables utilizadas por la implementación original y las nuevas características incorporadas para el modelo optimizado.
"""

columnas_originales_numericas = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

columnas_originales_categoricas = [
    "Sex", "Embarked"
]

columnas_numericas = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "Mother", "FarePerPerson"
]

columnas_categoricas = [
    "Sex", "Embarked", "Title", "Deck"
]

X_original = train[columnas_originales_numericas + columnas_originales_categoricas]
X = trabajo[columnas_numericas + columnas_categoricas]


"""###6. Separación Train, Validation y Test

Detecta tres conjuntos independientes y utiliza los mismos índices para ambos modelos.
De esta forma, la comparación entre el modelo original y el optimizado se realiza
sobre exactamente las mismas observaciones.

Primero separa el 80 % para entrenamiento y validación, y el 20 % para prueba.
Después divide el 80 % en 60 % para entrenamiento y 20 % para validación.
"""

indices = np.arange(len(train))

indices_train_val, indices_test = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

indices_train, indices_validation = train_test_split(
    indices_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y.iloc[indices_train_val]
)

# Conjuntos para el modelo original (se entrena con el 80 % y se evalúa en prueba).
X_train_val_original = X_original.iloc[indices_train_val]
X_test_original = X_original.iloc[indices_test]
y_train_val = y.iloc[indices_train_val]
y_test = y.iloc[indices_test]

# Conjuntos para el modelo optimizado (entrenamiento, validación y prueba).
X_train = X.iloc[indices_train]
X_validation = X.iloc[indices_validation]
X_train_val = X.iloc[indices_train_val]
X_test = X.iloc[indices_test]

y_train = y.iloc[indices_train]
y_validation = y.iloc[indices_validation]
y_test = y.iloc[indices_test]

print("\nSeparación de datos")
print("Entrenamiento (train):", X_train.shape)
print("Validación:", X_validation.shape)
print("Entrenamiento + Validación:", X_train_val.shape)
print("Prueba:", X_test.shape)


"""###7. Configuración del preprocesamiento

Identifica los valores faltantes en variables numéricas y categóricas.
"""

def crear_preprocesamiento(columnas_numericas_modelo, columnas_categoricas_modelo):
    return ColumnTransformer(
        transformers=[
            (
                "numericas",
                SimpleImputer(strategy="median"),
                columnas_numericas_modelo
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputacion", SimpleImputer(strategy="most_frequent")),
                        ("codificacion", OneHotEncoder(handle_unknown="ignore"))
                    ]
                ),
                columnas_categoricas_modelo
            )
        ]
    )


"""###8. Modelo original

Reproduce la configuración de la segunda entrega para establecer una línea base.

La configuración original tenía criterio Gini, profundidad máxima de 4,
min_samples_split de 30 y min_samples_leaf de 15.
"""

def crear_modelo_original():
    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocesamiento", crear_preprocesamiento(columnas_originales_numericas, columnas_originales_categoricas)),
            ("arbol", modelo)
        ]
    )


"""###9. Entrenamiento del modelo original

Entrena el modelo original con el 80 % de los datos (entrenamiento + validación) para reproducir las métricas de la entrega anterior.
"""

modelo_original = crear_modelo_original()
modelo_original.fit(X_train_val_original, y_train_val)


"""###10. Función de evaluación

Detecta las métricas necesarias para comparar los modelos.
"""

def evaluar_modelo(modelo, X_datos, y_datos):
    predicciones = modelo.predict(X_datos)

    return {
        "Accuracy": accuracy_score(y_datos, predicciones),
        "Precision": precision_score(y_datos, predicciones, zero_division=0),
        "Recall": recall_score(y_datos, predicciones, zero_division=0),
        "F1-score": f1_score(y_datos, predicciones, zero_division=0)
    }


"""###11. Evaluación inicial del modelo original
"""

resultados_original = {
    "Entrenamiento": evaluar_modelo(modelo_original, X_train_val_original, y_train_val),
    "Prueba": evaluar_modelo(modelo_original, X_test_original, y_test)
}

print("\nResultados del modelo original")
for conjunto, metricas in resultados_original.items():
    print(f"\n{conjunto}")
    for metrica, valor in metricas.items():
        print(f"{metrica:10s}: {valor:.4f}")


"""###12. Ajuste de hiperparámetros

Identifica diferentes configuraciones del árbol para buscar una mejora sin utilizar el conjunto de prueba durante la selección.

"""

configuraciones = []

for criterio in ["gini", "entropy"]:
    for profundidad in [4, 5, 6, 7, 8]:
        for minimo_split in [10, 15, 20, 30]:
            for minimo_hoja in [5, 6, 8, 10, 12, 15]:
                for peso in [None, "balanced"]:
                    configuraciones.append(
                        {
                            "criterion": criterio,
                            "max_depth": profundidad,
                            "min_samples_split": minimo_split,
                            "min_samples_leaf": minimo_hoja,
                            "class_weight": peso
                        }
                    )


"""###13. Selección de la mejor configuración

Identifica la mejor configuración utilizando primero F1-score y después Accuracy.
Cuando existen empates, selecciona la configuración más restrictiva entre las opcionesn empatadas mediante un mayor min_samples_split.
"""

resultados_busqueda = []

for configuracion in configuraciones:
    modelo_candidato = Pipeline(
        steps=[
            ("preprocesamiento", crear_preprocesamiento(columnas_numericas, columnas_categoricas)),
            (
                "arbol",
                DecisionTreeClassifier(
                    random_state=42,
                    **configuracion
                )
            )
        ]
    )

    modelo_candidato.fit(X_train, y_train)
    metricas_validacion = evaluar_modelo(
        modelo_candidato,
        X_validation,
        y_validation
    )

    resultados_busqueda.append(
        {
            **configuracion,
            **metricas_validacion,
            "modelo": modelo_candidato
        }
    )

resultados_busqueda.sort(
    key=lambda resultado: (
        resultado["F1-score"],
        resultado["Accuracy"],
        resultado["min_samples_split"]
    ),
    reverse=True
)

mejor_resultado = resultados_busqueda[0]

# Detecta una configuración equivalente en validación y utiliza mayor min_samples_split como desempate regularizador.
mejor_f1 = mejor_resultado["F1-score"]
mejor_accuracy = mejor_resultado["Accuracy"]
empates = [
    resultado
    for resultado in resultados_busqueda
    if np.isclose(resultado["F1-score"], mejor_f1)
    and np.isclose(resultado["Accuracy"], mejor_accuracy)
]

if len(empates) > 1:
    mejor_resultado = max(
        empates,
        key=lambda resultado: resultado["min_samples_split"]
    )

mejor_configuracion = {
    clave: mejor_resultado[clave]
    for clave in [
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "class_weight"
    ]
}

print("\nMejor configuración encontrada:")
print(mejor_configuracion)


"""###14. Entrenamiento del modelo optimizado

Entrena el modelo optimizado con el 80 % de los datos (entrenamiento + validación) utilizando la mejor configuración encontrada.
"""

modelo_optimizado = Pipeline(
    steps=[
        ("preprocesamiento", crear_preprocesamiento(columnas_numericas, columnas_categoricas)),
        (
            "arbol",
            DecisionTreeClassifier(
                random_state=42,
                **mejor_configuracion
            )
        )
    ]
)

modelo_optimizado.fit(X_train_val, y_train_val)


"""###15. Ajuste del umbral de decisión

Detecta el umbral que maximiza el F1-score sobre el conjunto de validación.
Esto permite mejorar el Recall sin sacrificar demasiado la Precision.
"""

probs_validation = modelo_optimizado.predict_proba(X_validation)[:, 1]

mejor_umbral = 0.5
mejor_f1_umbral = 0.0

for umbral in np.arange(0.10, 0.90, 0.02):
    predicciones = (probs_validation >= umbral).astype(int)
    f1 = f1_score(y_validation, predicciones, zero_division=0)
    if f1 > mejor_f1_umbral:
        mejor_f1_umbral = f1
        mejor_umbral = umbral

print(f"\nMejor umbral encontrado: {mejor_umbral:.2f} con F1 en validación: {mejor_f1_umbral:.4f}")


"""###16. Función de evaluación con umbral

Calcula las métricas usando un umbral de decisión personalizado.
"""

def evaluar_modelo_con_umbral(modelo, X_datos, y_datos, umbral):
    probs = modelo.predict_proba(X_datos)[:, 1]
    predicciones = (probs >= umbral).astype(int)

    return {
        "Accuracy": accuracy_score(y_datos, predicciones),
        "Precision": precision_score(y_datos, predicciones, zero_division=0),
        "Recall": recall_score(y_datos, predicciones, zero_division=0),
        "F1-score": f1_score(y_datos, predicciones, zero_division=0)
    }


"""###17. Evaluación del modelo optimizado

Calcula las métricas finales del modelo seleccionado en entrenamiento (80 %) y prueba (20 %), utilizando el umbral ajustado.
"""

resultados_optimizado = {
    "Entrenamiento": evaluar_modelo_con_umbral(
        modelo_optimizado, X_train_val, y_train_val, mejor_umbral
    ),
    "Prueba": evaluar_modelo_con_umbral(
        modelo_optimizado, X_test, y_test, mejor_umbral
    )
}

print("\nResultados del modelo optimizado")
for conjunto, metricas in resultados_optimizado.items():
    print(f"\n{conjunto}")
    for metrica, valor in metricas.items():
        print(f"{metrica:10s}: {valor:.4f}")


"""###18. Comparación de resultados

Identifica la diferencia entre el modelo original y el optimizado.
"""

print("\nComparación Final")

for conjunto in ["Entrenamiento", "Prueba"]:
    print(f"\n{conjunto}")
    for metrica in ["Accuracy", "Precision", "Recall", "F1-score"]:
        original = resultados_original[conjunto][metrica]
        optimizado = resultados_optimizado[conjunto][metrica]
        diferencia = optimizado - original
        print(
            f"{metrica:10s} | Original: {original:.4f} | "
            f"Optimizado: {optimizado:.4f} | "
            f"Cambio: {diferencia:+.4f}"
        )


"""###19. Matrices de confusión
"""

pred_original_test = modelo_original.predict(X_test_original)

probs_test_optimizado = modelo_optimizado.predict_proba(X_test)[:, 1]
pred_optimizado_test = (probs_test_optimizado >= mejor_umbral).astype(int)

matriz_original = confusion_matrix(y_test, pred_original_test)
matriz_optimizado = confusion_matrix(y_test, pred_optimizado_test)

print("\nMatriz de confusión - Modelo original")
print(matriz_original)

print("\nMatriz de confusión - Modelo optimizado")
print(matriz_optimizado)


"""###20. Diagnóstico de bias, varianza y ajuste

Calcula las diferencias entre entrenamiento y prueba para respaldar el diagnóstico.
"""

brecha_original = (
    resultados_original["Entrenamiento"]["Accuracy"]
    - resultados_original["Prueba"]["Accuracy"]
)

brecha_optimizado = (
    resultados_optimizado["Entrenamiento"]["Accuracy"]
    - resultados_optimizado["Prueba"]["Accuracy"]
)

print("\nBrecha Accuracy entrenamiento-prueba")
print(f"Original: {brecha_original:.4f}")
print(f"Optimizado: {brecha_optimizado:.4f}")


"""###21. Gráfica comparativa de métricas

Muestra las métricas obtenidas por los dos modelos en el conjunto de prueba.
"""

metricas = ["Accuracy", "Precision", "Recall", "F1-score"]
valores_original = [resultados_original["Prueba"][m] for m in metricas]
valores_optimizado = [resultados_optimizado["Prueba"][m] for m in metricas]

x_pos = np.arange(len(metricas))
ancho = 0.36

plt.figure(figsize=(10, 6))
plt.bar(x_pos - ancho / 2, valores_original, width=ancho, label="Original")
plt.bar(x_pos + ancho / 2, valores_optimizado, width=ancho, label="Optimizado")
plt.xticks(x_pos, metricas)
plt.ylim(0, 1)
plt.ylabel("Valor")
plt.title("Comparación de métricas en el conjunto de prueba")
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_metricas.png", dpi=300)
plt.show()


"""###22. Comparación de matrices de confusión
"""

fig, ejes = plt.subplots(1, 2, figsize=(11, 5))

for eje, matriz, titulo in [
    (ejes[0], matriz_original, "Modelo original"),
    (ejes[1], matriz_optimizado, "Modelo optimizado")
]:
    imagen = eje.imshow(matriz)
    fig.colorbar(imagen, ax=eje)
    eje.set_xticks([0, 1])
    eje.set_yticks([0, 1])
    eje.set_xticklabels(["No sobrevive", "Sobrevive"])
    eje.set_yticklabels(["No sobrevive", "Sobrevive"])
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Valor real")
    eje.set_title(titulo)

    for i in range(2):
        for j in range(2):
            eje.text(j, i, matriz[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("matrices_confusion_comparacion.png", dpi=300)
plt.show()


"""###23. Visualización del árbol optimizado
"""

modelo_arbol_optimizado = modelo_optimizado.named_steps["arbol"]
nombres_variables = (
    modelo_optimizado
    .named_steps["preprocesamiento"]
    .get_feature_names_out()
)

plt.figure(figsize=(20, 10))
plot_tree(
    modelo_arbol_optimizado,
    feature_names=nombres_variables,
    class_names=["No sobrevive", "Sobrevive"],
    filled=True,
    rounded=True,
    max_depth=7,
    fontsize=7
)
plt.title("Árbol de decisión optimizado")
plt.tight_layout()
plt.savefig("arbol_optimizado.png", dpi=300)
plt.show()


"""###24. Tabla final de resultados
"""

tabla_resultados = pd.DataFrame({
    "Métrica": metricas,
    "Original": valores_original,
    "Optimizado": valores_optimizado,
    "Mejora": np.array(valores_optimizado) - np.array(valores_original)
})

print("\nTabla final de resultados sobre TEST")
print(tabla_resultados.to_string(index=False))

