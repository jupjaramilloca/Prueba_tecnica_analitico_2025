# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- Equipo Vicepresidencia de Ecosistemas
-----------------------------------------------------------------------------
-- Fecha Creación: 20250203
-- Última Fecha Modificación: 20250203
-- Autores: juajaram
-- Últimos Autores: juajaram
-- Descripción:     Script de ejecución de los ETLs
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
"""
import os
import argparse
from datetime import datetime
from orquestador2.step import Step
from dateutil.relativedelta import relativedelta
import pandas as pd
import json
import pickle
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier



class LoadFiles(Step):
    """__DocString_ExtractTransformLoad__"""
    
    def fn_cargue_archivos(self, zona, prefijo, insumo):
        # Replace single quotes with double quotes
        insumo = insumo.replace("'", '"')  ## Reemplazar comillas simples por comillas dobles
        # Parse the JSON string into a dictionary
        try:
            sparky = self.getSparky()  ## Obtener objeto Sparky
            insumo_dict = json.loads(insumo)  ## Parsear la cadena JSON en un diccionario
            ruta = insumo_dict.get('ruta', None)  ## Obtener la ruta del diccionario
            ruta = ruta.encode('latin1').decode('utf-8')  ## Decodificar la ruta
            tbl = insumo_dict.get('tbl', None)  ## Obtener el nombre de la tabla del diccionario
            tbl = f'{zona}.{prefijo}_{tbl}'  ## Formatear el nombre de la tabla
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")  ## Imprimir error si falla el parseo del JSON
            return None

        try:
            print(f'Cargando {ruta}...')  ## Imprimir mensaje de carga
            if ruta.endswith('.xlsx'):
                # Leer un archivo Excel directamente en un DataFrame
                df_insumo = pd.read_excel(ruta)  ## Leer archivo Excel en un DataFrame
            elif ruta.endswith('.csv'):
                # Leer un archivo CSV directamente en un DataFrame
                df_insumo = pd.read_csv(ruta)  ## Leer archivo CSV en un DataFrame
            else:
                print('El archivo no es válido')  ## Imprimir mensaje si el archivo no es válido
                return

            # Reemplazar comas por barras verticales en el DataFrame
            df_insumo = df_insumo.replace(',', '|', regex=True)  ## Reemplazar comas por barras verticales
            df_insumo = df_insumo.replace('\n', ' ', regex=True)  ## Reemplazar saltos de línea por espacios

            # Subir el DataFrame al sistema Sparky y ejecutar una consulta SQL
            sparky.subir_df(df_insumo, nombre_tabla=f'{tbl}', modo='overwrite')  ## Subir DataFrame a Sparky
        except Exception as e:
            print(f'Error al cargar {tbl}: {str(e)}')  ## Imprimir error si falla la carga

    def ejecutar(self):
        """Método público no heredado que ejecuta el paso de la clase"""
        params = self.getGlobalConfiguration()  ## Obtener configuración global
        params.update(self.getStepConfig())  ## Actualizar con configuración del paso
        zona = params['zona_p']  ## Obtener zona
        prefijo = params['prefijo']  ## Obtener prefijo
        ##Insumo test
        self.fn_cargue_archivos(zona, prefijo, json.dumps(params['insumo_var_rpta_alt_oot']))  ## Cargar insumo de test
        # Insumo entrenamiento
        self.fn_cargue_archivos(zona, prefijo, json.dumps(params['var_rpta_trtest']))  ## Cargar insumo de entrenamiento
        # Insumos mestra cuotas pagos
        self.fn_cargue_archivos(zona, prefijo, json.dumps(params['maestra_cuotas_pagos']))  ## Cargar insumo de maestra cuotas pagos
        # Insumos master customer data
        self.fn_cargue_archivos(zona, prefijo, json.dumps(params['master_customer_data']))  ## Cargar insumo de master customer data
        # Insumos probabilidad obligaciones
        self.fn_cargue_archivos(zona, prefijo, json.dumps(params['probabilidad_oblig']))  ## Cargar insumo de probabilidad obligaciones
        None
        
class Entrenamiento(Step):
    def ejecutar(self):
        """Método público no heredado que ejecuta el paso de la clase"""
        params = self.getGlobalConfiguration()  ## Obtener configuración global
        params.update(self.getStepConfig())  ## Actualizar con configuración del paso
        # ejecucion de archivos sql del Step
        self.executeFolder(self.getSQLPath()+params["sql_folder"], params)  ## Ejecutar archivos SQL del paso
        None               

class Test(Step):
    def ejecutar(self):
        """Método público no heredado que ejecuta el paso de la clase"""
        params = self.getGlobalConfiguration()  ## Obtener configuración global
        params.update(self.getStepConfig())  ## Actualizar con configuración del paso
        # ejecucion de archivos sql del Step
        self.executeFolder(self.getSQLPath()+params["sql_folder"], params)  ## Ejecutar archivos SQL del paso
        None               


class Entrenamiento_modelo(Step):
    """Clase para el entrenamiento de modelos de clasificación"""

    def make_pipeline(self, estimator, numeric_features, categorical_features):
        """Crear un pipeline con un estimador dado"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())  ## Escalador para características numéricas
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  ## OneHotEncoder para características categóricas
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),  ## Transformador para características numéricas
                ('cat', categorical_transformer, categorical_features)  ## Transformador para características categóricas
            ])

        return Pipeline(
            [
                ('preprocessor', preprocessor),  ## Preprocesador
                ('variance_threshold', VarianceThreshold()),  ## Eliminar características constantes
                ('selectkbest', SelectKBest(score_func=f_classif)),  ## Selección de características
                ('estimator', estimator),  ## Estimador
            ]
        )

    def make_gridsearch(self, pipeline, param_grid, cv):
        """Crear un gridsearch con un pipeline y parámetros dados"""
        return GridSearchCV(
            estimator=pipeline,  ## Pipeline
            param_grid=param_grid,  ## Parámetros
            cv=cv,  ## Validación cruzada
        )

    def make_logistic_regressor(self, max_features, cv, numeric_features, categorical_features):
        """Crear un gridsearch con un pipeline y parámetros dados para Logistic Regression"""
        model = LogisticRegression()  ## Modelo de regresión logística
        param_grid = {
            "selectkbest__k": range(1, max_features + 1),  ## Rango de características a seleccionar
        }
        pipeline = self.make_pipeline(model, numeric_features, categorical_features)  ## Crear pipeline
        gridsearch = self.make_gridsearch(pipeline, param_grid, cv=cv)  ## Crear gridsearch
        return gridsearch

    def make_random_forest_classifier(self, max_features, cv, numeric_features, categorical_features):
        """Crear un gridsearch con un pipeline y parámetros dados para Random Forest"""
        model = RandomForestClassifier()  ## Modelo de Random Forest
        param_grid = {
            "selectkbest__k": range(1, max_features + 1),  ## Rango de características a seleccionar
            "estimator__n_estimators": [100, 200, 300],  ## Número de árboles
            "estimator__max_depth": [None, 10, 20, 30],  ## Profundidad máxima
        }
        pipeline = self.make_pipeline(model, numeric_features, categorical_features)  ## Crear pipeline
        gridsearch = self.make_gridsearch(pipeline, param_grid, cv=cv)  ## Crear gridsearch
        return gridsearch

    def make_decision_tree_classifier(self, max_features, cv, numeric_features, categorical_features):
        """Crear un gridsearch con un pipeline y parámetros dados para Decision Tree"""
        model = DecisionTreeClassifier()  ## Modelo de árbol de decisión
        param_grid = {
            "selectkbest__k": range(1, max_features + 1),  ## Rango de características a seleccionar
            "estimator__max_depth": [None, 10, 20, 30],  ## Profundidad máxima
        }
        pipeline = self.make_pipeline(model, numeric_features, categorical_features)  ## Crear pipeline
        gridsearch = self.make_gridsearch(pipeline, param_grid, cv=cv)  ## Crear gridsearch
        return gridsearch

    def make_xgboost_classifier(self, max_features, cv, numeric_features, categorical_features):
        """Crear un gridsearch con un pipeline y parámetros dados para XGBoost"""
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  ## Modelo XGBoost
        param_grid = {
            "selectkbest__k": range(1, max_features + 1),  ## Rango de características a seleccionar
            "estimator__max_depth": [3, 5, 7, 10],  ## Profundidad máxima
            "estimator__learning_rate": [0.01, 0.1, 0.2, 0.3],  ## Tasa de aprendizaje
            "estimator__n_estimators": [100, 200, 300]  ## Número de árboles
        }
        pipeline = self.make_pipeline(model, numeric_features, categorical_features)  ## Crear pipeline
        gridsearch = self.make_gridsearch(pipeline, param_grid, cv=cv)  ## Crear gridsearch
        return gridsearch

    def make_lightgbm_classifier(self, max_features, cv, numeric_features, categorical_features):
        """Crear un gridsearch con un pipeline y parámetros dados para LightGBM"""
        model = LGBMClassifier()  ## Modelo LightGBM
        param_grid = {
            "selectkbest__k": range(1, max_features + 1),  ## Rango de características a seleccionar
            "estimator__max_depth": [3, 5, 7, 10],  ## Profundidad máxima
            "estimator__learning_rate": [0.01, 0.1, 0.2, 0.3],  ## Tasa de aprendizaje
            "estimator__n_estimators": [100, 200, 300]  ## Número de árboles
        }
        pipeline = self.make_pipeline(model, numeric_features, categorical_features)  ## Crear pipeline
        gridsearch = self.make_gridsearch(pipeline, param_grid, cv=cv)  ## Crear gridsearch
        return gridsearch

    

    #### Métricas

    def compute_metrics(self, y_true, y_pred):
        """Evaluar el modelo usando accuracy, precision, recall y f1-score"""
        accuracy = accuracy_score(y_true, y_pred)  ## Calcular precisión
        precision = precision_score(y_true, y_pred, average='weighted')  ## Calcular precisión ponderada
        recall = recall_score(y_true, y_pred, average='weighted')  ## Calcular recall ponderado
        f1 = f1_score(y_true, y_pred, average='weighted')  ## Calcular F1-score ponderado
        return accuracy, precision, recall, f1

    def report_metrics(self, estimator, accuracy, precision, recall, f1):
        """Imprimir las métricas del modelo"""
        print(f"{estimator}:")  ## Imprimir nombre del estimador
        print(f"  Precisión: {accuracy}")  ## Imprimir precisión
        print(f"  Precisión: {precision}")  ## Imprimir precisión
        print(f"  Recall: {recall}")  ## Imprimir recall
        print(f"  F1 Score: {f1}")  ## Imprimir F1-score

    def load_model_from_disk(self, path_model, name_model):
        """Cargar el modelo desde el disco"""
        path = os.path.join(path_model, name_model)  ## Crear ruta completa
        if os.path.exists(path):  ## Verificar si el archivo existe
            with open(path, "rb") as file:  ## Abrir archivo en modo lectura binaria
                model = pickle.load(file)  ## Cargar modelo
        else:
            model = None  ## Si no existe, asignar None
        return model

    def save_model_to_disk(self, model, path_model, name_model):
        """Guardar el modelo en el disco"""
        path = os.path.join(path_model, name_model)  ## Crear ruta completa
        with open(path, "wb") as file:  ## Abrir archivo en modo escritura binaria
            pickle.dump(model, file)  ## Guardar modelo

    def compare_saved_model_with_new_model(self, saved_model, new_model, x_test, y_test):
        """Comparar modelos basándose en F1-score además de best_score_"""
        if saved_model is None:  ## Si no hay modelo guardado
            return new_model  ## Devolver nuevo modelo

        y_pred_new = new_model.predict(x_test)  ## Predicciones del nuevo modelo
        y_pred_saved = saved_model.predict(x_test)  ## Predicciones del modelo guardado

        f1_new = f1_score(y_test, y_pred_new, average='weighted')  ## Calcular F1-score del nuevo modelo
        f1_saved = f1_score(y_test, y_pred_saved, average='weighted')  ## Calcular F1-score del modelo guardado

        return new_model if f1_new > f1_saved else saved_model  ## Devolver el mejor modelo

    def load_train_and_test_datasets(self, df):
        """Cargar datos desde el datalake"""

        numeric_features =[
            'valor_cuota_mes',  ## Característica numérica
            'pago_total',  ## Característica numérica
            'total_ing',  ## Característica numérica
            'tot_activos',  ## Característica numérica
            'egresos_mes',  ## Característica numérica
            'tot_patrimonio',  ## Característica numérica
            'prob_propension',  ## Característica numérica
            'prob_alrt_temprana',  ## Característica numérica
            'prob_auto_cura'  ## Característica numérica
        ]

        categorical_features = [
            'marca_pago',  ## Característica categórica
            'tipo_cli',  ## Característica categórica
            'lote',  ## Característica categórica
            'segm',  ## Característica categórica
            'subsegm'  ## Característica categórica
        ]

        x = df[numeric_features + categorical_features]  ## Seleccionar características
        y = df["var_rpta_alt"]  ## Seleccionar variable objetivo

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  ## Dividir datos en entrenamiento y prueba

        return x_train, x_test, y_train, y_test, numeric_features, categorical_features  ## Devolver conjuntos de datos

    def get_args_from_command_line(self):
        """Obtener los argumentos desde la línea de comandos"""
        parser = argparse.ArgumentParser(description="Entrenamiento del Modelo de Clasificación")  ## Crear parser

        parser.add_argument(
            "--max_features",
            type=int,
            default=7,
            help="Número máximo de características a usar",  ## Argumento para número máximo de características
        )

        parser.add_argument(
            "--cv",
            type=int,
            default=5,
            help="Número de pliegues de validación cruzada",  ## Argumento para número de pliegues de validación cruzada
        )

        args = parser.parse_args()  ## Parsear argumentos
        return args

    def ejecutar(self):
        """Método público no heredado que ejecuta el paso de la clase"""
        params = self.getStepConfig()  ## Obtener configuración del paso
        ih = self.getHelper()  ## Obtener helper
        
        path_model = params['path_model'].encode('latin1').decode('utf-8')  ## Decodificar ruta del modelo
        name_model = params['name_model'].encode('latin1').decode('utf-8')  ## Decodificar nombre del modelo
        
        print("Iniciando el proceso de entrenamiento...")  ## Imprimir mensaje de inicio
        args = self.get_args_from_command_line()  ## Obtener argumentos de la línea de comandos
        print("Argumentos analizados exitosamente.")  ## Imprimir mensaje de éxito
        
        df = ih.obtener_dataframe(params['train_file_path'])  ## Obtener DataFrame de entrenamiento
        df = df.apply(lambda x: x.astype(str).str.encode('latin1').str.decode('utf-8'))  ## Decodificar DataFrame
        
        x_train, x_test, y_train, y_test, numeric_features, categorical_features = self.load_train_and_test_datasets(df)  ## Cargar conjuntos de datos
        print("Datos cargados exitosamente.")  ## Imprimir mensaje de éxito
        print(f"Forma de los datos de entrenamiento: {x_train.shape}")  ## Imprimir forma de los datos de entrenamiento
        print(f"Forma de los datos de prueba: {x_test.shape}")  ## Imprimir forma de los datos de prueba

        classifiers = [
            self.make_logistic_regressor(
                 max_features=args.max_features,  ## Número máximo de características
                 cv=args.cv,  ## Número de pliegues de validación cruzada
                 numeric_features=numeric_features,  ## Características numéricas
                 categorical_features=categorical_features,  ## Características categóricas
             ),
             self.make_random_forest_classifier(
                 max_features=args.max_features,  ## Número máximo de características
                 cv=args.cv,  ## Número de pliegues de validación cruzada
                 numeric_features=numeric_features,  ## Características numéricas
                categorical_features=categorical_features,  ## Características categóricas
             ),
             self.make_decision_tree_classifier(
                 max_features=args.max_features,  ## Número máximo de características
                 cv=args.cv,  ## Número de pliegues de validación cruzada
                 numeric_features=numeric_features,  ## Características numéricas
                 categorical_features=categorical_features,  ## Características categóricas
             ),
            #self.make_xgboost_classifier(
            #   max_features=args.max_features,  ## Número máximo de características
            #   cv=args.cv,  ## Número de pliegues de validación cruzada
            #   numeric_features=numeric_features,  ## Características numéricas
            #   categorical_features=categorical_features,  ## Características categóricas
            #),
             self.make_lightgbm_classifier(
                 max_features=args.max_features,  ## Número máximo de características
                 cv=args.cv,  ## Número de pliegues de validación cruzada
                 numeric_features=numeric_features,  ## Características numéricas
                 categorical_features=categorical_features,  ## Características categóricas
             ),
        ]

        for classifier in classifiers:
            print(f"Entrenando {classifier.estimator.__class__.__name__}...")  ## Imprimir mensaje de inicio de entrenamiento
            classifier.fit(x_train, y_train)  ## Entrenar clasificador
            print(f"Entrenamiento de {classifier.estimator.__class__.__name__} finalizado.")  ## Imprimir mensaje de fin de entrenamiento

        best_model = self.load_model_from_disk(path_model, name_model)  ## Cargar mejor modelo desde el disco

        for new_model in classifiers:
            best_model = self.compare_saved_model_with_new_model(best_model, new_model, x_test, y_test)  ## Comparar modelos
            self.save_model_to_disk(best_model, path_model, name_model)  ## Guardar mejor modelo en el disco

        y_test_pred = best_model.predict(x_test)  ## Predicciones del mejor modelo
        
        accuracy, precision, recall, f1 = self.compute_metrics(y_test, y_test_pred)  ## Calcular métricas
        self.report_metrics(best_model, accuracy, precision, recall, f1)  ## Reportar métricas
        print("Proceso de entrenamiento completado.")  ## Imprimir mensaje de fin de proceso
        print(f"Mejor modelo: {best_model.estimator.__class__.__name__}")  ## Imprimir nombre del mejor modelo
        print(f"F1 Score del mejor modelo: {f1}")  ## Imprimir F1-score del mejor modelo



class Inferencia(Step):
    """Clase para cargar el modelo de propensión de pago y realizar predicciones."""

    def cargar_modelo(self, model_path):
        """Carga el modelo desde la ruta especificada."""
        try:
            return joblib.load(model_path)  ## Carga el modelo desde la ruta especificada usando joblib
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo: {e}")  ## Lanza un error si no se puede cargar el modelo

    def obtener_mejor_modelo(self, loaded_model):
        """Obtiene el mejor estimador del modelo si fue entrenado con GridSearchCV."""
        return loaded_model.best_estimator_  ## Devuelve el mejor estimador del modelo si fue entrenado con GridSearchCV

    def verificar_encoder(self, best_model):
        """Verifica que el OneHotEncoder tenga la configuración correcta."""
        preprocessor = best_model.named_steps['preprocessor']  ## Obtiene el preprocesador del mejor modelo
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'cat':  ## Verifica si el transformador es categórico
                onehot_encoder = transformer.named_steps['onehot']  ## Obtiene el OneHotEncoder
                assert onehot_encoder.handle_unknown == 'ignore', (
                    "OneHotEncoder debe tener handle_unknown='ignore'"  ## Verifica que el OneHotEncoder maneje valores desconocidos
                )

    def preparar_datos(self, df_test):
        """Prepara los datos asegurando que tengan los tipos correctos y las columnas esperadas."""
        id_columns = df_test[['key', 'nit_enmascarado', 'num_oblig_orig_enmascarado', 
                              'num_oblig_enmascarado', 'fecha_var_rpta_alt']]  ## Selecciona las columnas de identificación
        
        df_test = df_test.astype({
            'valor_cuota_mes': 'float64',
            'pago_total': 'float64',
            'total_ing': 'float64',
            'tot_activos': 'float64',
            'egresos_mes': 'float64',
            'tot_patrimonio': 'float64',
            'prob_propension': 'float64',
            'prob_alrt_temprana': 'float64',
            'prob_auto_cura': 'float64',
            'marca_pago': 'object',
            'tipo_cli': 'object',
            'lote': 'object',
            'segm': 'object',
            'subsegm': 'object'
        })  ## Convierte las columnas a los tipos de datos correctos
        
        expected_columns = [
            'valor_cuota_mes', 'pago_total', 'total_ing', 'tot_activos', 'egresos_mes',
            'tot_patrimonio', 'prob_propension', 'prob_alrt_temprana', 'prob_auto_cura',
            'marca_pago', 'tipo_cli', 'lote', 'segm', 'subsegm'
        ]  ## Define las columnas esperadas
        
        df_test = df_test[expected_columns]  
        

        
        ## Selecciona las columnas esperadas
        return id_columns, df_test  ## Devuelve las columnas de identificación y los datos de prueba

    def predecir(self, best_model, df_test):
        """Realiza predicciones y obtiene las probabilidades sobre los datos de prueba."""
        id_columns, df_test = self.preparar_datos(df_test)  ## Prepara los datos de prueba
        
        predictions = best_model.predict(df_test)  ## Realiza predicciones sobre los datos de prueba
        
        # Verifica si el modelo tiene el método predict_proba
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(df_test)[:, 1]  ## Obtiene las probabilidades de la clase positiva
        else:
            probabilities = [None] * len(predictions)  ## En caso de que no haya predict_proba, asigna None
        
        df_test = pd.concat([id_columns, df_test], axis=1)  ## Combina las columnas de identificación con los datos de prueba
        df_test['predictions'] = predictions  ## Agrega las predicciones al DataFrame
        df_test['probability'] = probabilities  ## Agrega las probabilidades al DataFrame
        
        return df_test  ## Devuelve el DataFrame con las predicciones y probabilidades


    def ejecutar(self):
        """Método principal que ejecuta la secuencia completa de carga, verificación y predicción."""
        params = self.getStepConfig()  ## Obtiene la configuración del paso
        ih = self.getHelper()  ## Obtiene el helper
        sparky = self.getSparky()  ## Obtiene el objeto Sparky
        
        model_path = params['model_path'].encode('latin1').decode('utf-8')  ## Decodifica la ruta del modelo
        name_model = params['model_name'].encode('latin1').decode('utf-8')  ## Decodifica el nombre del modelo
        
        df_test = ih.obtener_dataframe(params['new_data'])  ## Obtiene el DataFrame de prueba
        numericas = [
            'valor_cuota_mes',
            'pago_total',
            'total_ing',
            'tot_activos',
            'egresos_mes',
            'tot_patrimonio',
            'prob_propension',
            'prob_alrt_temprana',
            'prob_auto_cura'

        ]
        
        categoricas = ['marca_pago', 'tipo_cli', 'lote', 'segm', 'subsegm']
        
        # Reemplazar NaN en columnas numéricas con la mediana

        df_test[numericas] = df_test[numericas].apply(lambda x: x.fillna(x.median()))
        
        # Reemplazar NaN en columnas categóricas con la moda

        df_test[categoricas] = df_test[categoricas].apply(lambda x: x.fillna(x.mode()[0]))
        
        loaded_model = self.cargar_modelo(os.path.join(model_path, name_model))  ## Carga el modelo desde el disco
        best_model = self.obtener_mejor_modelo(loaded_model)  ## Obtiene el mejor modelo
        self.verificar_encoder(best_model)  ## Verifica el OneHotEncoder
        df = self.predecir(best_model, df_test)  ## Realiza predicciones sobre los datos de prueba
        
        sparky.subir_df(df, 
                        nombre_tabla=params['output'],
                        modo='overwrite')  ## Sube el DataFrame con las predicciones al sistema Sparky
        
        
class Guardar_informacion(Step):
    """__DocString_ExtractTransformLoad__"""

    def ejecutar(self):
        params = self.getStepConfig()  ## Obtener configuración del paso
        helper = self.getHelper()  ## Obtener helper
        
        self.executeFolder(self.getSQLPath()+params["sql_folder"], params)  ## Ejecutar archivos SQL del paso
        
        df_1 = helper.obtener_dataframe(params['sumision'])  ## Obtener DataFrame de sumisión
        df_2 = helper.obtener_dataframe(params['resultado'])  ## Obtener DataFrame de resultado
        
        ruta_resultado = os.path.join(os.getcwd(), '002_resultados')  ## Crear ruta para guardar resultados
        
        df_1.to_csv(os.path.join(ruta_resultado, params['name_file_1']), index=False)  ## Guardar DataFrame de sumisión como CSV
        df_2.to_csv(os.path.join(ruta_resultado, params['name_file_2']), index=False)  ## Guardar DataFrame de resultado como CSV
