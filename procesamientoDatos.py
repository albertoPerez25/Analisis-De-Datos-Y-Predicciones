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

# --- 2. Definición de Variables ---
target = 'fallo'

# Asumimos que las columnas temporales (hora, día, mes) NO están aún creadas.
# Si tienes una columna de timestamp, deberías extraerlas primero. Aah pero esto no esta hecho con IA que desis
numerical_features_base = ['temperatura', 'presion', 'sensor_ruido', 'sensor_3']
categorical_features_base = ['modo_operacion', 'operador']

'''
NO HACE FALTA
# Verifica que las columnas existen en el DataFrame
actual_numerical_features = numerical_features_base#[col for col in numerical_features_base if col in df_train.columns]
actual_categorical_features = categorical_features_base#[col for col in categorical_features_base if col in df_train.columns]
'''

print(f"\nCaracterísticas numéricas identificadas: {numerical_features_base}")
print(f"Características categóricas identificadas: {categorical_features_base}")
print(f"Variable objetivo: {target}")


# Manejo específico de NaNs en 'operador' (según análisis previo)
# Crea una copia para no modificar el original directamente en cada iteración
df_processed = df_train.copy()
'''
# Redundante porque ya lo hace abajo tambien
df_processed['operador'] = df_processed['operador'].fillna('Desconocido')
print("\nValores NaN en 'operador' reemplazados por 'Desconocido'.")
# Verifica si hay NaNs en otras columnas
print("\nNaNs restantes:")
print(df_processed.isnull().sum())
'''

# Separar X e y
X = df_processed.drop(target, axis=1)
y = df_processed[target]

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
'''
ESTO NO HACE FALTA
# Asegurarse que existen
promising_numerical = [f for f in promising_numerical if f in actual_numerical_features]
promising_categorical = [f for f in promising_categorical if f in actual_categorical_features]
'''

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

# --- 8. Bucle de Experimentación ---
def compararModelos():
    results = []
    '''
    INTENTO MIO DE HACER LO QUE HACE GEMINI CON UN SCORE NORMAL (NO VA PORQUE DA ERROR EN MANTENIMIENTO)

    for set_name, set_config in feature_sets.items():
        print(f"\nEvaluando Feature Set: {set_name}")
        current_preprocessor = set_config['preprocessor']
        # Seleccionamos solo las columnas necesarias para este set ANTES de pasarlo al pipeline
        current_X = X[set_config['features']].copy()

        for nombre,modelo in models.items():
            print(f"\nEvaluando modelo: {nombre}")
            if isinstance(current_preprocessor, Pipeline): # Caso Set 3
                # El modelo se añade como último paso al pipeline existente
                full_pipeline = Pipeline(steps=current_preprocessor.steps + [('model', modelo)])
            else: # Casos Set 1 y Set 2
                # El preprocesador es un ColumnTransformer
                full_pipeline = Pipeline(steps=[('preprocess', current_preprocessor),
                                            ('model', modelo)])
            modelo.fit(current_X, y)
            results.append({'Model': nombre, 'Score': modelo.score(current_X, y)})


    print("\n--- Resultados Iniciales ---",results)
    results = []
    '''

    print(f"\n--- Iniciando Experimentación con {n_splits}-Fold Cross-Validation ---")
    print(f"Métrica de evaluación: {scoring_metric}")

    '''
    # Lo que habia hecho Gemini: 
    # (Basicamente calcula para cada modelo y para cada configuracion distintos
    # scores con distintos puntos de corte en X para el train y el test, 
    # y procede a hacer la media que es el output)
    '''
    for set_name, set_config in feature_sets.items():
        print(f"\nEvaluando Feature Set: {set_name}")
        current_preprocessor = set_config['preprocessor']
        # Seleccionamos solo las columnas necesarias para este set ANTES de pasarlo al pipeline
        current_X = X[set_config['features']].copy()

        for model_name, model in models.items():
            # Crear el pipeline completo: Preprocesador (puede incluir PolyFeatures) + Modelo
            # Si el preprocesador ya es un pipeline (como en Set 3), esto anida pipelines
            if isinstance(current_preprocessor, Pipeline): # Caso Set 3
                # El modelo se añade como último paso al pipeline existente
                full_pipeline = Pipeline(steps=current_preprocessor.steps + [('model', model)])
            else: # Casos Set 1 y Set 2
                # El preprocesador es un ColumnTransformer
                full_pipeline = Pipeline(steps=[('preprocess', current_preprocessor),
                                                ('model', model)])

            try:
                # Realizar Validación Cruzada
                scores = cross_val_score(full_pipeline, current_X, y,
                                        cv=cv, scoring=scoring_metric, n_jobs=-1) # n_jobs=-1 usa todos los cores

                results.append({
                    'Feature Set': set_name,
                    'Model': model_name,
                    f'Mean {scoring_metric.upper()}': np.mean(scores),
                    f'Std Dev {scoring_metric.upper()}': np.std(scores),
                    'Scores per Fold': scores
                })
                print(f"  {model_name}: Mean F1 = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})") # Con formato bonito y todo wow

            except Exception as e:
                print(f"  ERROR ejecutando {model_name} con {set_name}: {e}")
                results.append({
                    'Feature Set': set_name,
                    'Model': model_name,
                    f'Mean {scoring_metric.upper()}': np.nan,
                    f'Std Dev {scoring_metric.upper()}': np.nan,
                    'Scores per Fold': [np.nan] * n_splits
                })
    return results
            
def mejorGradientBoosting():
    set_name = 'Todas_Mas_Interacciones'
    set_config = feature_sets[set_name]

    current_preprocessor = set_config['preprocessor']
    current_X = X[set_config['features']].copy()

    model_name = "GradientBoosting"
    model = GradientBoostingClassifier(random_state=42, n_estimators=400)

    full_pipeline = Pipeline(steps=current_preprocessor.steps + [('model', model)])

    return validacionCruzada(model_name,full_pipeline,current_X, set_name)

def mejorRandomForest():
    set_name = 'Todas_Mas_Interacciones'
    set_config = feature_sets[set_name]

    current_preprocessor = set_config['preprocessor']
    current_X = X[set_config['features']].copy()

    model_name = "RandomForest"
    model = RandomForestClassifier(random_state=42, n_estimators=550, class_weight='balanced', n_jobs=-1)

    full_pipeline = Pipeline(steps=current_preprocessor.steps + [('model', model)])

    return validacionCruzada(model_name,full_pipeline,current_X, set_name)

def validacionCruzada(model_name,pipeline,X,set_name):
    results = []
    print("Ejecutando validación cruzada de",model_name,"...")
    # Realizar Validación Cruzada
    scores = cross_val_score(pipeline, X, y,
                            cv=cv, scoring=scoring_metric, n_jobs=-1) # n_jobs=-1 usa todos los cores

    results.append({
        'Feature Set': set_name,
        'Model': model_name,
        f'Mean {scoring_metric.upper()}': np.mean(scores),
        f'Std Dev {scoring_metric.upper()}': np.std(scores),
        'Scores per Fold': scores
    })

    return results

results = mejorGradientBoosting()
print(results)
# --- 9. Presentación de Resultados ---
results_df = pd.DataFrame(results)
print("\n--- Resumen de Resultados ---")
# Ordenar por la métrica media descendente para ver los mejores primero
results_df = results_df.sort_values(by=f'Mean {scoring_metric.upper()}', ascending=False)
print(results_df[[ 'Feature Set', 'Model', f'Mean {scoring_metric.upper()}', f'Std Dev {scoring_metric.upper()}']])

print("\n--- Fin de la Experimentación ---")

