import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

class CafeMLModel:
    def __init__(self):
        self.sales_model = None
        self.scaler = StandardScaler()
        
    def train_sales_prediction(self, df):
        """Entrena modelo de predicción de ventas"""
        try:
            print("🤖 Iniciando entrenamiento del modelo de ventas...")
            
            # Agrupar ventas por día
            daily_sales = df.groupby(df['fecha'].dt.date).agg({
                'total': 'sum',
                'cantidad': 'sum'
            }).reset_index()
            daily_sales.columns = ['fecha', 'ventas_totales', 'productos_vendidos']
            daily_sales['fecha'] = pd.to_datetime(daily_sales['fecha'])
            
            print(f"📊 Días con ventas: {len(daily_sales)}")
            print(f"📅 Rango de fechas: {daily_sales['fecha'].min()} a {daily_sales['fecha'].max()}")
            
            if len(daily_sales) < 5:
                return {
                    "error": "Insuficientes datos para entrenar el modelo",
                    "dias_disponibles": len(daily_sales)
                }
            
            # Crear características temporales
            daily_sales = daily_sales.sort_values('fecha')
            daily_sales['dia_semana'] = daily_sales['fecha'].dt.dayofweek
            daily_sales['dia_mes'] = daily_sales['fecha'].dt.day
            daily_sales['mes'] = daily_sales['fecha'].dt.month
            
            # Crear lags (ventas de días anteriores)
            for lag in [1, 2, 3]:
                daily_sales[f'ventas_lag_{lag}'] = daily_sales['ventas_totales'].shift(lag)
            
            # Eliminar filas con NaN
            daily_sales = daily_sales.dropna()
            
            if len(daily_sales) < 3:
                return {
                    "error": "Datos insuficientes después de limpieza",
                    "muestras_validas": len(daily_sales)
                }
            
            # Preparar datos para entrenamiento
            features = ['dia_semana', 'dia_mes', 'mes', 'ventas_lag_1', 'ventas_lag_2', 'ventas_lag_3']
            X = daily_sales[features]
            y = daily_sales['ventas_totales']
            
            print(f"🎯 Entrenando con {len(X)} muestras y {len(features)} características")
            
            # Entrenar modelo
            self.sales_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.sales_model.fit(X, y)
            
            # Evaluar modelo
            y_pred = self.sales_model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # Importancia de características
            feature_importance = dict(zip(features, self.sales_model.feature_importances_))
            
            results = {
                "precision_r2": round(r2, 3),
                "error_mae": round(mae, 2),
                "muestras_entrenamiento": len(X),
                "importancia_caracteristicas": feature_importance,
                "ventas_promedio": round(y.mean(), 2)
            }
            
            print(f"✅ Modelo entrenado - R²: {r2:.3f}, MAE: ${mae:.2f}")
            print(f"📈 Ventas promedio: ${y.mean():.2f}")
            print("🔍 Importancia de características:")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"   {feature}: {importance:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            return {"error": str(e)}
    
    def predict_sales(self, df, days=7):
        """Genera predicciones de ventas para los próximos días"""
        try:
            print(f"🔮 Generando predicciones para {days} días...")
            
            if self.sales_model is None:
                model_result = self.train_sales_prediction(df)
                if "error" in model_result:
                    return [{"error": model_result["error"]}]
            
            # Obtener datos históricos
            daily_sales = df.groupby(df['fecha'].dt.date)['total'].sum().reset_index()
            daily_sales.columns = ['fecha', 'ventas_totales']
            daily_sales['fecha'] = pd.to_datetime(daily_sales['fecha'])
            daily_sales = daily_sales.sort_values('fecha')
            
            # Preparar última fila para predicción
            last_date = daily_sales['fecha'].max()
            last_sales = daily_sales['ventas_totales'].tail(3).tolist()
            
            predictions = []
            current_date = last_date
            
            for i in range(days):
                current_date += timedelta(days=1)
                
                # Crear características para la predicción
                features = {
                    'dia_semana': current_date.weekday(),
                    'dia_mes': current_date.day,
                    'mes': current_date.month,
                    'ventas_lag_1': last_sales[-1] if len(last_sales) >= 1 else daily_sales['ventas_totales'].mean(),
                    'ventas_lag_2': last_sales[-2] if len(last_sales) >= 2 else daily_sales['ventas_totales'].mean(),
                    'ventas_lag_3': last_sales[-3] if len(last_sales) >= 3 else daily_sales['ventas_totales'].mean()
                }
                
                # Hacer predicción
                X_pred = pd.DataFrame([features])
                prediction = self.sales_model.predict(X_pred)[0]
                
                # Asegurar que la predicción no sea negativa
                prediction = max(0, prediction)
                
                predictions.append({
                    "fecha": current_date.strftime('%Y-%m-%d'),
                    "prediccion": round(float(prediction), 2),
                    "dia_semana": current_date.strftime('%A')
                })
                
                # Actualizar last_sales para siguiente predicción
                last_sales.append(prediction)
                if len(last_sales) > 3:
                    last_sales.pop(0)
            
            print(f"✅ {len(predictions)} predicciones generadas exitosamente")
            for pred in predictions:
                print(f"   📅 {pred['fecha']} ({pred['dia_semana']}): ${pred['prediccion']}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error en predicciones: {e}")
            return [{"error": str(e)}]
    
    def analyze_product_performance(self, df):
        """Analiza el rendimiento de productos"""
        try:
            print("📈 Analizando rendimiento de productos...")
            
            product_performance = df.groupby('producto_nombre').agg({
                'cantidad': 'sum',
                'sub_total': 'sum',
                'precio': 'mean',
                'costo': 'mean'
            }).reset_index()
            
            product_performance.columns = ['producto', 'unidades_vendidas', 'ingresos_totales', 'precio_promedio', 'costo_promedio']
            
            # Calcular métricas adicionales
            product_performance['margen'] = product_performance['precio_promedio'] - product_performance['costo_promedio']
            product_performance['rentabilidad'] = (product_performance['margen'] / product_performance['costo_promedio']) * 100
            product_performance['participacion_mercado'] = (product_performance['ingresos_totales'] / product_performance['ingresos_totales'].sum()) * 100
            
            # Ordenar por ingresos
            product_performance = product_performance.sort_values('ingresos_totales', ascending=False)
            
            print(f"✅ Análisis completado para {len(product_performance)} productos")
            
            return product_performance.to_dict('records')
            
        except Exception as e:
            print(f"❌ Error analizando productos: {e}")
            return [{"error": str(e)}]
    
    def create_sales_plots(self, df):
        """Crea gráficos de ventas"""
        try:
            print("📊 Generando gráficos...")
            
            plots = {}
            
            # 1. Ventas diarias
            daily_sales = df.groupby(df['fecha'].dt.date)['total'].sum().reset_index()
            daily_sales.columns = ['fecha', 'ventas_totales']
            daily_sales = daily_sales.sort_values('fecha')
            
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily_sales['fecha'],
                y=daily_sales['ventas_totales'],
                mode='lines+markers',
                name='Ventas Diarias',
                line=dict(color='#8B4513')
            ))
            fig_daily.update_layout(
                title='Ventas Diarias',
                xaxis_title='Fecha',
                yaxis_title='Ventas ($)',
                template='plotly_white'
            )
            plots['daily_sales'] = {
                'data': fig_daily.to_dict()['data'],
                'layout': fig_daily.to_dict()['layout']
            }
            
            # 2. Ventas por producto
            product_sales = df.groupby('producto_nombre')['cantidad'].sum().nlargest(10)
            fig_products = go.Figure()
            fig_products.add_trace(go.Bar(
                x=product_sales.index,
                y=product_sales.values,
                marker_color='#D2691E'
            ))
            fig_products.update_layout(
                title='Top 10 Productos Más Vendidos',
                xaxis_title='Producto',
                yaxis_title='Unidades Vendidas',
                template='plotly_white'
            )
            plots['product_sales'] = {
                'data': fig_products.to_dict()['data'],
                'layout': fig_products.to_dict()['layout']
            }
            
            # 3. Ventas por día de la semana
            df['dia_semana'] = df['fecha'].dt.day_name()
            weekday_sales = df.groupby('dia_semana')['total'].sum()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_sales = weekday_sales.reindex(weekday_order)
            
            fig_weekday = go.Figure()
            fig_weekday.add_trace(go.Bar(
                x=weekday_sales.index,
                y=weekday_sales.values,
                marker_color='#A0522D'
            ))
            fig_weekday.update_layout(
                title='Ventas por Día de la Semana',
                xaxis_title='Día',
                yaxis_title='Ventas ($)',
                template='plotly_white'
            )
            plots['weekday_sales'] = {
                'data': fig_weekday.to_dict()['data'],
                'layout': fig_weekday.to_dict()['layout']
            }
            
            print("✅ Gráficos generados exitosamente")
            return plots
            
        except Exception as e:
            print(f"❌ Error creando gráficos: {e}")
            return {}
    
    def detect_anomalies(self, df):
        """Detección de anomalías en transacciones basada en datos reales"""
        try:
            print("🚨 Detectando anomalías en transacciones...")
            
            if df.empty:
                return []
            
            # Calcular estadísticas por transacción
            mean_total = df['total'].mean()
            std_total = df['total'].std()
            
            # Identificar anomalías (valores fuera de 2 desviaciones estándar)
            anomalies = []
            for idx, row in df.iterrows():
                z_score = abs(row['total'] - mean_total) / std_total if std_total > 0 else 0
                
                if z_score > 2:  # Más de 2 desviaciones estándar
                    anomalies.append({
                        "fecha": row['fecha'].strftime('%Y-%m-%d') if hasattr(row['fecha'], 'strftime') else str(row['fecha']),
                        "transaccion_id": f"T-{idx}",
                        "valor": round(float(row['total']), 2),
                        "es_anomalia": True,
                        "puntuacion_anomalia": round(float(z_score), 2),
                        "producto": row.get('producto_nombre', 'N/A')
                    })
            
            print(f"✅ Detección de anomalías completada: {len(anomalies)} encontradas")
            return anomalies[:10]  # Limitar a 10 anomalías
            
        except Exception as e:
            print(f"❌ Error en detección de anomalías: {e}")
            return []
    
    def generate_business_recommendations(self, df_sales, df_inventory=None, df_products=None):
        """Genera recomendaciones basadas en análisis de datos reales"""
        try:
            print("💡 Generando recomendaciones de negocio...")
            
            recommendations = []
            
            if df_sales.empty:
                return recommendations
            
            # 1. Recomendación basada en productos más vendidos
            top_products = df_sales.groupby('producto_nombre')['cantidad'].sum().nlargest(3)
            if len(top_products) > 0:
                top_product = top_products.index[0]
                recommendations.append({
                    "titulo": f"Optimizar stock de {top_product}",
                    "descripcion": f"Es el producto más vendido con {int(top_products.iloc[0])} unidades",
                    "impacto": "Alto - Asegurar disponibilidad continua",
                    "prioridad": "alta"
                })
            
            # 2. Recomendación basada en días de mayor venta
            daily_sales = df_sales.groupby(df_sales['fecha'].dt.day_name())['total'].sum()
            if len(daily_sales) > 0:
                best_day = daily_sales.idxmax()
                recommendations.append({
                    "titulo": f"Reforzar personal los {best_day}s",
                    "descripcion": f"Los {best_day}s generan las mayores ventas (${daily_sales.max():.2f})",
                    "impacto": "Medio - Mejorar servicio en picos",
                    "prioridad": "media"
                })
            
            # 3. Recomendación basada en margen de productos
            if 'margen' in df_sales.columns:
                profitable_products = df_sales.groupby('producto_nombre')['margen'].sum().nlargest(2)
                if len(profitable_products) > 0:
                    profitable_product = profitable_products.index[0]
                    recommendations.append({
                        "titulo": f"Promocionar {profitable_product}",
                        "descripcion": f"Genera el mayor margen de ganancia (${profitable_products.iloc[0]:.2f})",
                        "impacto": "Alto - Incrementar rentabilidad",
                        "prioridad": "alta"
                    })
            
            # 4. Recomendación basada en tendencia temporal
            if len(df_sales) > 7:
                last_week = df_sales['total'].tail(7).sum()
                previous_week = df_sales['total'].head(7).sum() if len(df_sales) > 14 else last_week
                trend = "creciente" if last_week > previous_week else "decreciente"
                recommendations.append({
                    "titulo": f"Tendencia de ventas {trend}",
                    "descripcion": f"Ventas de la última semana: ${last_week:.2f} vs anterior: ${previous_week:.2f}",
                    "impacto": "Media - Monitorear tendencia",
                    "prioridad": "media"
                })
            
            print(f"✅ {len(recommendations)} recomendaciones generadas")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generando recomendaciones: {e}")
            return []

# Funciones de serialización
def convert_to_serializable(obj):
    """Convierte objetos numpy/pandas a tipos serializables para JSON"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

def serialize_ml_results(results):
    """Serializa los resultados de ML para la API"""
    if isinstance(results, dict):
        return {key: serialize_ml_results(value) for key, value in results.items()}
    elif isinstance(results, list):
        return [serialize_ml_results(item) for item in results]
    else:
        return convert_to_serializable(results)

# Instancia global del modelo
ml_model = CafeMLModel()