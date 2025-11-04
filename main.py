from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sys
import os
from typing import Dict, Any, List
import json
import traceback
from datetime import datetime

# Configurar path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'app'))

print(f"🔍 Directorio actual: {current_dir}")

# Importar módulos
try:
    from database import get_sales_data, get_inventory_data, get_product_performance
    from ml_model import ml_model, serialize_ml_results
    print("✅ Módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    # Funciones dummy como fallback
    def get_sales_data(): return pd.DataFrame()
    def get_inventory_data(): return pd.DataFrame()
    def get_product_performance(): return pd.DataFrame()
    class DummyMLModel:
        def train_sales_prediction(self, df): return {"status": "modelo no disponible"}
        def predict_sales(self, df): return []
        def analyze_product_performance(self, df): return []
        def detect_anomalies(self, df): return []
        def generate_business_recommendations(self, df1, df2, df3): return []
        def create_sales_plots(self, df): return {}
    ml_model = DummyMLModel()
    def serialize_ml_results(obj): return obj

app = FastAPI(title="Estación Café ML API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar templates y archivos estáticos
try:
    templates = Jinja2Templates(directory="app/templates")
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    print("✅ Templates y static files configurados")
except Exception as e:
    print(f"⚠️  Error configurando templates/static: {e}")
    templates = Jinja2Templates(directory=".")
    print("⚠️  Usando directorio actual para templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Página principal del dashboard"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"""
        <html>
            <body>
                <h1>Estación Café ML Dashboard</h1>
                <p>Error cargando template: {e}</p>
                <p>El sistema está funcionando pero hay problemas con los templates.</p>
                <p><a href="/api/docs">Ver documentación de la API</a></p>
            </body>
        </html>
        """)

@app.get("/api/sales-data")
async def api_get_sales_data():
    """Obtiene datos de ventas y análisis completo con ML"""
    try:
        print("🔍 Cargando datos de ventas...")
        df = get_sales_data()
        
        if not hasattr(df, 'empty'):
            return JSONResponse(
                status_code=500,
                content={"error": f"Error: get_sales_data() retornó {type(df)} en lugar de DataFrame"}
            )
        
        if df.empty:
            print("⚠️ No hay datos de ventas en la base de datos")
            return JSONResponse(
                status_code=404,
                content={"error": "No hay datos de ventas disponibles en la base de datos"}
            )
        
        print("🤖 Ejecutando modelos de ML...")
        # Entrenar modelo de predicción
        model_results = ml_model.train_sales_prediction(df)
        
        # Generar predicciones
        predictions = ml_model.predict_sales(df)
        
        # Análisis de productos
        product_analysis = ml_model.analyze_product_performance(df)
        
        # Detección de anomalías
        anomalies = ml_model.detect_anomalies(df)
        
        # Recomendaciones
        df_inventory = get_inventory_data()
        df_products = get_product_performance()
        recommendations = ml_model.generate_business_recommendations(df, df_inventory, df_products)
        
        # Gráficos
        plots = ml_model.create_sales_plots(df)
        
        # Estadísticas generales
        stats = {
            "total_ventas": round(df['total'].sum(), 2),
            "ventas_promedio": round(df['total'].mean(), 2),
            "producto_mas_vendido": df.groupby('producto_nombre')['cantidad'].sum().idxmax() if not df.empty else "No disponible",
            "dias_analizados": df['fecha'].nunique(),
            "total_transacciones": len(df),
            "margen_promedio": round(df['margen'].mean(), 2) if 'margen' in df.columns else 0,
            "rentabilidad_promedio": round(df['rentabilidad'].mean(), 2) if 'rentabilidad' in df.columns else 0
        }
        
        response_data = {
            "estadisticas": stats,
            "modelo_prediccion": serialize_ml_results(model_results),
            "predicciones": serialize_ml_results(predictions),
            "analisis_productos": serialize_ml_results(product_analysis),
            "anomalias_detectadas": serialize_ml_results(anomalies),
            "recomendaciones": serialize_ml_results(recommendations),
            "visualizaciones": serialize_ml_results(plots)
        }
        
        print("✅ Datos preparados para enviar a la API")
        return response_data
        
    except Exception as e:
        print(f"❌ Error en get_sales_data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error procesando datos: {str(e)}"}
        )

@app.get("/api/inventory-analysis")
async def api_get_inventory_analysis():
    """Análisis de inventario con datos reales"""
    try:
        print("📦 Cargando datos de inventario...")
        df = get_inventory_data()
        
        if not hasattr(df, 'empty'):
            return JSONResponse(
                status_code=500,
                content={"error": f"get_inventory_data() retornó {type(df)} en lugar de DataFrame"}
            )
        
        if df.empty:
            print("⚠️ No hay datos de inventario en la base de datos")
            return JSONResponse(
                status_code=404,
                content={"error": "No hay datos de inventario disponibles"}
            )
        
        # Análisis básico con datos reales
        total_inventario = (df['stock_actual'] * df['costo']).sum()
        productos_bajo_stock = df[df['stock_actual'] < 10]
        
        response_data = {
            "resumen_inventario": {
                "valor_total_inventario": round(total_inventario, 2),
                "total_productos": len(df),
                "productos_bajo_stock": len(productos_bajo_stock),
                "producto_mas_caro": df.loc[df['costo'].idxmax()]['consumible'] if len(df) > 0 else "N/A",
                "categoria_mayor_stock": df.groupby('tipo_consumible')['stock_actual'].sum().idxmax() if len(df) > 0 else "N/A"
            },
            "productos_criticos": serialize_ml_results(productos_bajo_stock[['consumible', 'stock_actual', 'costo', 'tipo_consumible']].to_dict('records')),
            "resumen_categorias": serialize_ml_results(df.groupby('tipo_consumible').agg({
                'stock_actual': 'sum',
                'costo': 'mean',
                'consumible': 'count'
            }).rename(columns={'consumible': 'cantidad_productos'}).reset_index().to_dict('records'))
        }
        
        print("✅ Inventario analizado correctamente")
        return response_data
        
    except Exception as e:
        print(f"❌ Error en inventory-analysis: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analizando inventario: {str(e)}"}
        )

@app.get("/api/product-performance")
async def api_get_product_performance():
    """Rendimiento de productos con datos reales"""
    try:
        print("🏆 Cargando rendimiento de productos...")
        df = get_product_performance()
        
        if not hasattr(df, 'empty'):
            return JSONResponse(
                status_code=500,
                content={"error": f"get_product_performance() retornó {type(df)} en lugar de DataFrame"}
            )
        
        if df.empty:
            print("⚠️ No hay datos de rendimiento en la base de datos")
            return JSONResponse(
                status_code=404,
                content={"error": "No hay datos de rendimiento de productos disponibles"}
            )
            
        response_data = serialize_ml_results(df.to_dict('records'))
        print("✅ Rendimiento de productos cargado")
        return response_data
        
    except Exception as e:
        print(f"❌ Error en product-performance: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error obteniendo rendimiento: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar conexión a DB"""
    try:
        print("🏥 Verificando salud del sistema...")
        df_sales = get_sales_data()
        df_inventory = get_inventory_data()
        df_products = get_product_performance()
        
        # Verificar que todos son DataFrames
        errors = []
        if not hasattr(df_sales, 'empty'):
            errors.append(f"sales_data: {type(df_sales)}")
        if not hasattr(df_inventory, 'empty'):
            errors.append(f"inventory_data: {type(df_inventory)}")
        if not hasattr(df_products, 'empty'):
            errors.append(f"products_data: {type(df_products)}")
        
        if errors:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "unhealthy",
                    "database_connection": "error",
                    "error": f"Funciones retornaron tipos incorrectos: {', '.join(errors)}",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Verificar estructura básica de datos
        data_check = {
            "sales_data": {
                "has_data": not df_sales.empty,
                "records": len(df_sales),
                "columns": list(df_sales.columns) if not df_sales.empty else []
            },
            "inventory_data": {
                "has_data": not df_inventory.empty,
                "records": len(df_inventory),
                "columns": list(df_inventory.columns) if not df_inventory.empty else []
            },
            "products_data": {
                "has_data": not df_products.empty,
                "records": len(df_products),
                "columns": list(df_products.columns) if not df_products.empty else []
            }
        }
        
        response = {
            "status": "healthy",
            "database_connection": "ok",
            "data_check": data_check,
            "timestamp": datetime.now().isoformat()
        }
        
        print("✅ Health check completado")
        return response
        
    except Exception as e:
        print(f"❌ Health check falló: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy", 
                "database_connection": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/ml/predict-sales")
async def predict_sales(days: int = 7):
    """Endpoint específico para predicciones de ventas"""
    try:
        df = get_sales_data()
        
        if not hasattr(df, 'empty'):
            return JSONResponse(
                status_code=500,
                content={"error": f"get_sales_data() retornó {type(df)} en lugar de DataFrame"}
            )
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"error": "No hay datos para realizar predicciones"}
            )
        
        predictions = ml_model.predict_sales(df, days)
        return {
            "periodo_prediccion": f"{days} días",
            "predicciones": serialize_ml_results(predictions),
            "datos_entrenamiento": {
                "dias_historicos": df['fecha'].nunique(),
                "rango_fechas": {
                    "inicio": df['fecha'].min().isoformat() if not df.empty else None,
                    "fin": df['fecha'].max().isoformat() if not df.empty else None
                }
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error en predicción: {str(e)}"}
        )

@app.get("/api/ml/anomaly-detection")
async def detect_sales_anomalies():
    """Detección de anomalías en ventas"""
    try:
        df = get_sales_data()
        
        if not hasattr(df, 'empty'):
            return JSONResponse(
                status_code=500,
                content={"error": f"get_sales_data() retornó {type(df)} en lugar de DataFrame"}
            )
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"error": "No hay datos para detección de anomalías"}
            )
        
        anomalies = ml_model.detect_anomalies(df)
        return {
            "anomalias_detectadas": anomalies,
            "resumen": {
                "total_transacciones": len(df),
                "anomalias_encontradas": len(anomalies),
                "tasa_anomalias": f"{(len(anomalies) / len(df) * 100):.1f}%" if len(df) > 0 else "0%"
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error en detección de anomalías: {str(e)}"}
        )

@app.get("/api/ml/recommendations")
async def get_business_recommendations():
    """Recomendaciones de negocio"""
    try:
        df_sales = get_sales_data()
        df_inventory = get_inventory_data()
        df_products = get_product_performance()
        
        if not hasattr(df_sales, 'empty') or not hasattr(df_inventory, 'empty') or not hasattr(df_products, 'empty'):
            return JSONResponse(
                status_code=500,
                content={"error": "Error en tipos de datos retornados por las funciones de database"}
            )
        
        if df_sales.empty:
            return JSONResponse(
                status_code=404,
                content={"error": "No hay datos suficientes para generar recomendaciones"}
            )
        
        recommendations = ml_model.generate_business_recommendations(
            df_sales, df_inventory, df_products
        )
        
        return {
            "recomendaciones_negocio": recommendations,
            "base_datos": {
                "ventas_analizadas": len(df_sales),
                "productos_inventario": len(df_inventory),
                "productos_rendimiento": len(df_products)
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generando recomendaciones: {str(e)}"}
        )

@app.get("/api/docs")
async def api_documentation():
    """Documentación de los endpoints disponibles"""
    return {
        "endpoints": {
            "health": {"method": "GET", "url": "/health", "description": "Verifica el estado del sistema y conexión a BD"},
            "sales_data": {"method": "GET", "url": "/api/sales-data", "description": "Datos completos de ventas con todos los análisis ML"},
            "inventory": {"method": "GET", "url": "/api/inventory-analysis", "description": "Análisis de inventario con datos reales"},
            "products": {"method": "GET", "url": "/api/product-performance", "description": "Rendimiento de productos con métricas reales"},
            "predictions": {"method": "GET", "url": "/api/ml/predict-sales?days=7", "description": "Predicciones de ventas basadas en datos históricos"},
            "anomalies": {"method": "GET", "url": "/api/ml/anomaly-detection", "description": "Detección de transacciones anómalas"},
            "recommendations": {"method": "GET", "url": "/api/ml/recommendations", "description": "Recomendaciones basadas en análisis de datos"}
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor FastAPI en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")