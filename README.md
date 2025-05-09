# opeco modelo opcion pagos

Modelo de opcin de pagos, etiqueta prospectos clientes para ofrecer opciones de pagos a las obligaciones

---

[[_TOC_]]

---
## Instalación

Se recomienda el uso de python 3.9.12 (https://www.python.org/ftp/python/3.9.12/python-3.9.12-amd64.exe)

Se recomienda realizar la instalación en un [**ambiente virtual**]

*Si se tiene configurado Artifactory:*
```
pip install veco-opeco-modelo-opcion-pagos
```
*Si no se tiene configurado Artifactory:*

```
pip install veco-opeco-modelo-opcion-pagos -i https://artifactory.apps.bancolombia.com/api/pypi/pypi-bancolombia/simple --trusted-host artifactory.apps.bancolombia.com
```

**NOTA:**
El generador de paquetes ```veco-opeco-generatorpkg``` trae consigo el ejecutable ```.PyPipEnv.bat```. Este ejecutable instala en un ambiente virtual, dentro del directorio del proyecto, todo lo necesario para la ejecución del flujo *-python_requires: especificado en el archivo ```setup.cfg```-*.

---
## Ejecución

Si se trabaja con un ambiente virtual, se debe activar primero. [**Más información sobre el uso de ambientes virtuales**]

*Se debe ejecutar el siguiente comando:*
```
python -m veco_opeco_modelo_opcion_pagos.ejecucion
```

Para efectos de la generación de logs para la calendarización se pueden indicar los parámetros directamente en los siguientes comandos.

*Logs de estabilidad:*
```
python -m veco_opeco_modelo_opcion_pagos.ejecucion -lt "est"
```

*Logs de compilación:*
```
python -m veco_opeco_modelo_opcion_pagos.ejecucion -lt "cmp" -pl [porcentaje]
```

En estos comandos para la calendarización el parámetro ```lt``` hace referencia al tipo de log, estabilidad ```est``` o compilación ```cmp```. Cabe resaltar que, si se va a generar un log de compilación, se requiere también el parámetro ```pl``` que hace referencia al porcentaje límite de datos que se toma de las tablas insumo para dicha ejecución (valor entero entre 1 y 100). De igual manera, si se habilita tanto el log de estabilidad o el de compilación, la carpeta para almacenar los logs generados será ```logs_calendarizacion``` y para otros casos la carpeta será ```logs```; en ambos casos se creará la carpeta en el directorio de trabajo actual.

Adicionalmente, con el fin de abreviar los comandos de ejecución, se habilitó utilizar el nombre del paquete (con guiones bajos) directamente para reemplazar la sintáxis de módulos de python; lo cuál permite sustituir la expresión ```python -m veco_opeco_modelo_opcion_pagos.ejecucion``` por ```veco_opeco_modelo_opcion_pagos``` en cada comando de ejecución si así lo desea.

*Fecha de ingestión:*
```
python -m veco_opeco_modelo_opcion_pagos.ejecucion -idate 20240228
```

También es posible usar el comando ```idate``` para ingresar fecha de ingestión. El formato de entrada es ```AAAAMMDD``` y dentro del flujo toma como tipo de dato ```datetime.datetime```. Por defecto toma la fecha actual de ejecución.

Este parámetro es utilizable durante toda la rutina invocando ```self.kwa["idate"]```, siempre que esté dentro de un método de la clase que herede de ```Step```.

**NOTA:**
El generador de paquetes ```veco-opeco-generatorpkg``` trae consigo el ejecutable ```PyRun.bat```. Este ejecutable activa el ambiente virtual dentro del directorio del proyecto y ejecuta el flujo. **No** tiene configurado parámetros iniciales.

---
## Prerrequisitos

El paquete ha sido generado para la versión de Python
	```
    3.9.12
    ```
. Las librerías o paquetes necesarios para la ejecución son:
- `openpyxl>=3.1.2`
- `master-validation>=2.5.3`
- `pyodbc==4.0.27`
- `orquestador2>=1.2.2`
- `veco-opeco-respaldologs>=1.1.3`

---
## Insumos y resultados

Los insumos utilizados en el proceso son:

| Zona de Datos | Tabla |
| - | - |
| _zone_x_ | _table_y_ |
| _zone_z_ | _table_k_ |

Los resultados obtenidos son:

| Zona de Datos | Tabla | Descripción | Tipo de Ingestión |
| - | - | - | - |
| _zone_results_ | _table_n_ | Esta información debe describir la tabla. | Incremental |
| _zone_results_ | _table_h_ | Esta información debe describir la tabla. | Full |

---