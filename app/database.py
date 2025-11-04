from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
import time

# CONEXIÓN A POSTGRESQL EN DOCKER
DATABASE_URL = "postgresql://admin:estacionPass2025@localhost:5555/estacioncafedb"

print(f"🔗 Conectando a PostgreSQL en Docker: postgresql://admin:****@localhost:5555/estacioncafedb")

# Configurar engine con parámetros corregidos
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"connect_timeout": 10}  # Solo connect_timeout es válido
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_database_connection():
    """Testea la conexión a la base de datos con diagnóstico completo"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with engine.connect() as connection:
                # Test básico
                result = connection.execute(text("SELECT version();"))
                db_version = result.fetchone()[0]
                print(f"✅ Conexión exitosa a PostgreSQL: {db_version.split(',')[0]}")
                
                # Verificar tablas existentes
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
                tables_result = connection.execute(text(tables_query))
                tables = [row[0] for row in tables_result]
                print(f"📊 Tablas en la base de datos: {tables}")
                
                # Verificar datos en tablas clave
                key_tables = ['bills', 'bill_details', 'products', 'Consumable']
                for table in key_tables:
                    if table in tables:
                        count_query = f'SELECT COUNT(*) as count FROM "{table}";'
                        count_result = connection.execute(text(count_query))
                        count = count_result.fetchone()[0]
                        print(f"   {table}: {count} registros")
                    else:
                        print(f"   {table}: NO EXISTE")
                
                return True
                
        except Exception as e:
            print(f"❌ Intento {attempt + 1}/{max_retries} - Error de conexión: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5
                print(f"🔄 Reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
    
    print("💥 No se pudo establecer conexión con la base de datos")
    return False

def get_sales_data():
    """Obtiene datos de ventas para el análisis"""
    try:
        print("🔍 Obteniendo datos de ventas desde PostgreSQL...")
        
        if not test_database_connection():
            return pd.DataFrame()
        
        # Consulta optimizada para obtener datos de ventas
        query = """
        SELECT 
            b.bill_id,
            b.date as fecha_factura,
            b.total,
            bd.quantity as cantidad,
            bd.sub_total,
            p.name as producto_nombre,
            p.price as precio,
            p.cost as costo,
            EXTRACT(DOW FROM b.date) as dia_semana,
            EXTRACT(MONTH FROM b.date) as mes
        FROM bills b
        JOIN bill_details bd ON b.bill_id = bd.bill_id
        JOIN products p ON bd.product_id = p.product_id
        WHERE b.date IS NOT NULL
        ORDER BY b.date
        """
        
        df = pd.read_sql_query(query, engine)
        print(f"📈 Datos de ventas obtenidos: {len(df)} registros")
        
        if not df.empty:
            # Procesar datos
            df['fecha'] = pd.to_datetime(df['fecha_factura'])
            df['margen'] = df['precio'] - df['costo']
            df['rentabilidad'] = (df['margen'] / df['costo'].replace(0, 1)) * 100
            
            print(f"📅 Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
            print(f"💰 Ventas totales: ${df['total'].sum():.2f}")
            print(f"📦 Productos únicos: {df['producto_nombre'].nunique()}")
            print(f"🛒 Transacciones totales: {df['bill_id'].nunique()}")
        else:
            print("⚠️ La consulta de ventas no retornó datos")
        
        return df
        
    except Exception as e:
        print(f"❌ Error en get_sales_data: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def get_inventory_data():
    """Obtiene datos de inventario"""
    try:
        print("📦 Obteniendo datos de inventario desde PostgreSQL...")
        
        if not test_database_connection():
            return pd.DataFrame()
            
        query = """
        SELECT 
            c.name as consumible,
            c.quantity as stock_actual,
            c.cost as costo,
            ct.name as tipo_consumible,
            s.name as proveedor
        FROM "Consumable" c
        JOIN consumable_type ct ON c.consumable_type_id = ct.consumable_type_id
        JOIN suppliers s ON c.supplier_id = s.supplier_id
        WHERE c.active = true
        ORDER BY c.quantity ASC
        """
        df = pd.read_sql_query(query, engine)
        print(f"📦 Datos de inventario obtenidos: {len(df)} consumibles")
        
        if not df.empty:
            total_valor = (df['stock_actual'] * df['costo']).sum()
            bajo_stock = len(df[df['stock_actual'] < 10])
            print(f"💰 Valor total inventario: ${total_valor:.2f}")
            print(f"⚠️  Productos con stock bajo: {bajo_stock}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error en get_inventory_data: {e}")
        return pd.DataFrame()

def get_product_performance():
    """Obtiene rendimiento de productos"""
    try:
        print("🏆 Obteniendo rendimiento de productos desde PostgreSQL...")
        
        if not test_database_connection():
            return pd.DataFrame()
            
        query = """
        SELECT 
            p.name as producto,
            COUNT(bd.bill_details_id) as veces_vendido,
            SUM(bd.quantity) as total_vendido,
            SUM(bd.sub_total) as ingresos_totales,
            AVG(p.price) as precio_promedio,
            AVG(p.cost) as costo_promedio,
            (AVG(p.price) - AVG(p.cost)) as margen_promedio
        FROM products p
        LEFT JOIN bill_details bd ON p.product_id = bd.product_id
        WHERE p.active = true
        GROUP BY p.name
        HAVING COUNT(bd.bill_details_id) > 0
        ORDER BY ingresos_totales DESC
        """
        df = pd.read_sql_query(query, engine)
        print(f"🏆 Datos de rendimiento obtenidos: {len(df)} productos con ventas")
        
        if not df.empty:
            print(f"📈 Producto más vendido: {df.iloc[0]['producto']}")
            print(f"💰 Ingresos totales: ${df['ingresos_totales'].sum():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error en get_product_performance: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("🧪 Ejecutando prueba de conexión a la base de datos...")
    test_database_connection()