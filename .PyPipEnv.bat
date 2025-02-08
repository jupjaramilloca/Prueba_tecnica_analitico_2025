@echo off
echo Localizando la carpeta de usuario local...
set "carpeta_usuario=%USERPROFILE%"

echo Validando carpeta pip...
set "carpeta_pip=%carpeta_usuario%\pip"
if not exist "%carpeta_pip%" (
    mkdir "%carpeta_pip%"
    ( echo [global]
    echo index-url=https://artifactory.apps.bancolombia.com/api/pypi/pypi-bancolombia/simple
    echo trusted-host=artifactory.apps.bancolombia.com ) > "%carpeta_pip%\pip.ini"
    echo Carpeta con archivo .ini creada en: %carpeta_pip%\pip.ini
)

echo Creando ambiente virtual...
python -m venv .venv

echo Activando ambiente virtual...
call .\.venv\Scripts\activate

echo Instalando dependencias...
python.exe -m pip install --upgrade pip
pip install --no-cache-dir -e.

pause