import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Para manejar NaNs numéricos si los hubiera

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB # Requiere datos densos (OHE está bien)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Nota: Para Random Forest y Gradient Boosting, librerías como XGBoost o LightGBM
# suelen ofrecer mejor rendimiento y más control, pero empezamos con scikit-learn.

# --- 1. Carga de Datos ---
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que 'train.csv' esté en el directorio correcto.")
    exit()
try:
    df_test = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que 'test.csv' esté en el directorio correcto.")
    exit()

# --- 2. Definición de Variables ---
target = 'fallo'

# Asumimos que las columnas temporales (hora, día, mes) NO están aún creadas.
# Si tienes una columna de timestamp, deberías extraerlas primero. Aah pero esto no esta hecho con IA que desis
numerical_features_base = ['temperatura', 'presion', 'sensor_ruido', 'sensor_3']
categorical_features_base = ['modo_operacion', 'operador']

print(f"\nCaracterísticas numéricas identificadas: {numerical_features_base}")
print(f"Características categóricas identificadas: {categorical_features_base}")
print(f"Variable objetivo: {target}")


# Manejo específico de NaNs en 'operador' (según análisis previo)
# Crea una copia para no modificar el original directamente en cada iteración
df_processed_train = df_train.copy()
X_test = df_test.copy()

# Separar X e y
X_train = df_processed_train.drop(target, axis=1)
y_train = df_processed_train[target]

# --- 4. Definición de Pipelines de Preprocesamiento ---

# Pipeline para variables numéricas:
# - Quitar NaNs con la mediana (más robusto a outliers)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Pipeline para variables categóricas:
# - Quitar NaNs con una constante
# - Aplicar One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')), # Por si hay otros NaNs categóricos
    ('onehot', OneHotEncoder()) # (handle_unknown='ignore', sparse_output=False) handle_unknown para robustez en CV
    # Como no hay demasiada cardinalidad, onehot debería de ser el mejor, pero podemos probar otros
])

# --- 5. Definición de Conjuntos de Características (Feature Sets) ---

feature_sets = {}

# Set 1: Todas las variables originales
preprocessor_all = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_base),
        ('cat', categorical_transformer, categorical_features_base)
    ],
    remainder='drop' # No incluir otras columnas si las hubiera
)
feature_sets['Todas_Originales'] = {
    'features': numerical_features_base + categorical_features_base,
    'preprocessor': preprocessor_all
}
print("\nDefinido Set 1: Todas las características originales")

# Set 2: Variables más prometedoras (Quitamos 'sensor_ruido')
promising_numerical = ['temperatura', 'presion', 'sensor_3']
promising_categorical = ['modo_operacion', 'operador'] # Mantenemos las categóricas prometedoras

preprocessor_promising = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, promising_numerical),
        ('cat', categorical_transformer, promising_categorical)
    ],
    remainder='drop'
)
feature_sets['Prometedoras'] = {
    'features': promising_numerical + promising_categorical,
    'preprocessor': preprocessor_promising
}
print("Definido Set 2: Características prometedoras")


# Set 3: Todas las originales + Interacciones (Polinómicas de grado 2 solo entre numéricas)
# Creamos un pipeline que primero preprocesa y luego añade interacciones

# Preprocesador solo para obtener las numéricas y las categóricas codificadas
preprocessor_base_interactions = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_base),
        ('cat', categorical_transformer, categorical_features_base)
    ],
    remainder='passthrough' # Mantenemos otras columnas (aunque no debería haber)
)

# Necesitamos saber cuántas columnas numéricas escaladas hay para aplicar PolyFeatures
# Esto es un poco más complejo dentro de un pipeline general.
# Una forma es aplicar PolyFeatures *después* del ColumnTransformer completo.
# Advertencia: Esto creará interacciones entre dummies categóricas, lo cual no siempre es deseado.
# Una aproximación más controlada sería crear un pipeline específico.

# Alternativa más simple: Crear un pipeline que incluya PolyFeatures al final
# Aplicará PolyFeatures a TODO lo que salga del ColumnTransformer (numéricas escaladas + categóricas OHE)
pipeline_with_poly = Pipeline([
    ('preprocess', preprocessor_all), # Usa el preprocesador del Set 1
    ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
    # interaction_only=False crea x^2, y^2, x*y
    # include_bias=False evita añadir una columna de unos
])
# Para este set, el 'preprocessor' es todo el pipeline hasta PolyFeatures
feature_sets['Todas_Mas_Interacciones'] = {
    'features': numerical_features_base + categorical_features_base, # Features originales que entran
    'preprocessor': pipeline_with_poly # El preprocesador AHORA incluye PolyFeatures
}
print("Definido Set 3: Todas las características + Interacciones Polinómicas (Aplicado a todo post-procesamiento inicial)")
print("Nota: El Set 3 crea interacciones entre *todas* las variables preprocesadas (incluyendo dummies).")

# --- 6. Definición de Modelos a Probar ---
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'), # random_state es una semilla para los random. class_weight para desbalanceo aplicando pesos según aparición en cada clase
    "GaussianNB": GaussianNB(), # No tiene class_weight, más sensible a desbalanceo
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1), # n_estimators es numero de árboles .n_jobs=-1 usa todos los cores
    "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=100) # GB de sklearn no tiene class_weight directo
}

# --- 7. Configuración de Validación Cruzada ---
n_splits = 5 # Número de folds
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# Usamos 'f1' como métrica. Para binario es f1 por defecto de la clase positiva (1)
# Si fuera multiclase o necesitaras otra media, usarías make_scorer(f1_score, average='weighted') por ejemplo.
scoring_metric = 'f1'

def mejorGradientBoosting():
    set_name = 'Todas_Mas_Interacciones'
    set_config = feature_sets[set_name]

    current_preprocessor = set_config['preprocessor']
    current_X_train = X_train[set_config['features']].copy()
    current_X_test = X_test[set_config['features']].copy()

    model_name = "GradientBoosting"
    model = GradientBoostingClassifier(random_state=42, n_estimators=400)

    full_pipeline = Pipeline(steps=current_preprocessor.steps + [('model', model)])

    print("Ejecutando fit de",model_name,"...")
    #full_pipeline.fit(current_X_train[:len(current_X_train)//2], y_train[:len(current_X_train)//2])
    full_pipeline.fit(current_X_train, y_train)
    #train_score = full_pipeline.score(current_X_train[len(current_X_train)//2:], y_train[len(current_X_train)//2:])
    test_predict = full_pipeline.predict(current_X_test)

    return test_predict

# Dividir el dataframe en filas, y pasarselas al predict de uno en uno
# Escribir el resultado del predict en el formato correcto
# Imprimir por pantalla la estimación de terminación


csv_prueba = 'pruebaPredict.csv'
columns = ['id', 'fallo']

results = pd.DataFrame(columns=columns)
set_name = 'Todas_Mas_Interacciones'
set_config = feature_sets[set_name]

current_preprocessor = set_config['preprocessor']
current_X_train = X_train[set_config['features']].copy()
current_X_test = X_test[set_config['features']].copy()

model_name = "GradientBoosting"
model = GradientBoostingClassifier(random_state=42, n_estimators=400)

full_pipeline = Pipeline(steps=current_preprocessor.steps + [('model', model)])

print("Ejecutando fit de",model_name,"...")
#full_pipeline.fit(current_X_train[:len(current_X_train)//2], y_train[:len(current_X_train)//2])
full_pipeline.fit(current_X_train, y_train)

        
for i in range(1, len(current_X_test), 500):
    test_predict = full_pipeline.predict(current_X_test[i:i+500])
    

    indi = [j for j in range(i, i+500,1)]
    valor = [test_predict[k] for k in range(499)]

    results = pd.DataFrame(list(zip(indi,valor)), columns=columns)

    results.to_csv(csv_prueba, mode='a', header=False, index=False)