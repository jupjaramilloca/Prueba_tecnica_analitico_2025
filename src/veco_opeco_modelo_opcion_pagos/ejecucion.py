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
-- Descripción:     Script de ejecución de la rutina
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
"""
import os
from datetime import datetime
from orquestador2.orquestador2 import Orchestrator
from master_validation.master_val import Master_val
from veco_opeco_respaldologs import SaveLogFile, StandardExecutionArgs
from veco_opeco_modelo_opcion_pagos.etl import (
    LoadFiles,Entrenamiento,Test,Entrenamiento_modelo,Inferencia,Guardar_informacion
)


class RunOrchestador:
    """Clase que inicializa el orquestador"""

    def main(self):
        # Fecha/hora inicio ejecución
        start = datetime.now()
        package_name = "opeco modelo opcion pagos"
        # Setear argumentos de entrada para el flujo
        sea = StandardExecutionArgs(descripcion=package_name)
        kw = sea.execution_arguments()
        # Crear carpeta de logs según necesidad
        if not os.path.exists(kw["log_path"]):
            print("Se creó carpeta de logs, según necesidad, ya que no existía.")
            os.mkdir(kw["log_path"])
        # Pasos del flujo a ejecutar en orden indicado
        steps = [
            #LoadFiles(**kw),
            # Entrenamiento(**kw),
            # Test(**kw),
            Entrenamiento_modelo(**kw),
            Inferencia(**kw),
            Guardar_informacion(**kw)
        ]
        # Instanciar orquestador y ejecutar
        orquestador = Orchestrator(package_name, steps, **kw)
        # Instanciar Maestro de Validaciones
        mv = Master_val(
            zona_tabla_input=orquestador.globalConfig["zona_tabla_input_mv"],
            zona_tabla_output=orquestador.globalConfig["zona_tabla_output_mv"],
            sparky=orquestador.sparky,
            activar=False # Activar o desactivar Maestro de Validaciones
        )
        # Instanciar orquestador y ejecutar
        excepcion = ""
        try:
            if mv.activar:
                if mv.insumo():
                    if mv.ejecucion(orquestador):
                        if not mv.razonabilidad():
                            excepcion = "No se aprobó el control de razonabilidad del Maestro de Validaciones"
                    else:
                        excepcion = "No se aprobó el control de ejecución del Maestro de Validaciones"
                else:
                    excepcion = "No se aprobó el control de insumos del Maestro de Validaciones"
            else:
                orquestador.ejecutar()
        except Exception as e:
            excepcion = str(e)
        finally:
            # Instanciar el respaldo de logs
            slf = SaveLogFile(
                active=False # Activar o desactivar respaldo de logs
            )
            slf.save_log_file(orquestador, start, excepcion)

def main():
    # Captura desde consola argumentos iniciales
    run_orquestador = RunOrchestador()
    run_orquestador.main()

if __name__ == "__main__":
    main()
