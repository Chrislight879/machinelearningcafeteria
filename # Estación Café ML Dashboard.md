# Estación Café ML Dashboard

## Instalación del Proyecto

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd machinelearningcafeteria
```

### 2. Crear y activar entorno virtual

#### En Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### En Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python -c "import fastapi; import pandas; import sklearn; print('Instalación exitosa!')"
```

## Dependencias Principales

El proyecto utiliza las siguientes dependencias principales:

- FastAPI v0.104.1 - Framework web
- Uvicorn v0.24.0 - Servidor ASGI
- Pandas v2.1.2 - Análisis de datos
- SQLAlchemy v2.0.23 - ORM para base de datos
- Scikit-learn v1.3.2 - Machine Learning
- Plotly v5.18.0 - Visualización de datos
- Python-jose v3.3.0 - Autenticación JWT
- Passlib v1.7.4 - Manejo de contraseñas
- Jinja2 v3.1.2 - Motor de plantillas
- Python-dotenv v1.0.0 - Variables de entorno

## Ejecutar la aplicación

```bash
python main.py
```

La aplicación estará disponible en: http://localhost:8000

## Solución de problemas comunes

Si encuentras el error "ModuleNotFoundError", asegúrate de:

1. Tener el entorno virtual activado
2. Haber instalado todas las dependencias:
```bash
pip install -r requirements.txt
```

Si hay problemas con psycopg2:
```bash
# En Ubuntu/Debian:
sudo apt-get install python3-dev libpq-dev
pip install psycopg2-binary

# En Fedora:
sudo dnf install python3-devel libpq-devel
pip install psycopg2-binary

# En macOS con Homebrew:
brew install postgresql
pip install psycopg2-binary
```