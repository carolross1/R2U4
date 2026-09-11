"""
Generador de dataset sintético — Habilidades Blandas
Extracción de Conocimientos en Base de Datos | Unidad IV | Recuperación 2

Genera 5000 registros de evaluación de habilidades blandas para empleados
ficticios, con base en la investigación previa (4 habilidades, cada una
segmentada en subcolumnas ponderadas que suman 1.0):

  1. Comunicación      -> com_*  (5 subcolumnas)
  2. Colaboración       -> col_*  (4 subcolumnas)
  3. Liderazgo          -> lid_*  (5 subcolumnas)
  4. Resolución de Problemas -> res_* (5 subcolumnas)

Cada subcolumna se califica en escala 0-10. El *_score de cada habilidad
es el promedio ponderado de sus subcolumnas según los pesos justificados
en la investigación.

Uso:
    python generar_dataset.py
Genera: datos/dataset_habilidades_blandas.csv (5000 filas)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = None # Cambiado a None para que genere datos diferentes cada vez
N = 5000
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------
# 1. Pesos por habilidad (documentados y justificados en la investigación)
# ---------------------------------------------------------------------
PESOS = {
    "com": {
        "com_claridad_expresion": 0.25,
        "com_escucha_activa": 0.25,
        "com_comunicacion_escrita": 0.20,
        "com_retroalimentacion": 0.15,
        "com_adaptacion_audiencia": 0.15,
    },
    "col": {
        "col_trabajo_equipo": 0.30,
        "col_cumplimiento_compromisos": 0.25,
        "col_disposicion_ayudar": 0.20,
        "col_contribucion_grupal": 0.25,
    },
    "lid": {
        "lid_toma_decisiones": 0.25,
        "lid_motivacion_equipo": 0.25,
        "lid_delegacion": 0.20,
        "lid_responsabilidad": 0.15,
        "lid_vision_estrategica": 0.15,
    },
    "res": {
        "res_analisis_situacion": 0.25,
        "res_pensamiento_critico": 0.25,
        "res_creatividad_soluciones": 0.20,
        "res_efectividad_solucion": 0.20,
        "res_tiempo_resolucion": 0.10,
    },
}

HABILIDADES = {
    "com": "Comunicación",
    "col": "Colaboración",
    "lid": "Liderazgo",
    "res": "Resolución de Problemas",
}

# ---------------------------------------------------------------------
# 2. Catálogo de áreas / departamentos (para poder segmentar estadística)
# ---------------------------------------------------------------------
AREAS_DEPTOS = {
    "Ventas": ["Ventas Nacionales", "Ventas Internacionales", "Atención a Clientes"],
    "Marketing": ["Marketing Digital", "Investigación de Mercado", "Comunicación de Marca"],
    "Operaciones": ["Logística", "Producción", "Control de Calidad"],
    "Tecnología": ["Desarrollo", "Soporte Técnico", "Infraestructura"],
    "Recursos Humanos": ["Reclutamiento", "Capacitación", "Nómina"],
    "Finanzas": ["Contabilidad", "Tesorería", "Auditoría Interna"],
}
AREAS = list(AREAS_DEPTOS.keys())

NOMBRES = ["María", "José", "Juan", "Ana", "Luis", "Laura", "Carlos", "Sofía",
           "Miguel", "Daniela", "Jorge", "Fernanda", "Ricardo", "Paola", "Diego",
           "Andrea", "Alejandro", "Karla", "Roberto", "Valeria", "Francisco",
           "Gabriela", "Sergio", "Mariana", "Pedro", "Camila", "Raúl", "Ximena"]
APELLIDOS = ["García", "Martínez", "López", "Hernández", "González", "Pérez",
             "Sánchez", "Ramírez", "Cruz", "Flores", "Rodríguez", "Morales",
             "Reyes", "Jiménez", "Torres", "Vázquez", "Ortiz", "Gómez",
             "Castillo", "Romero"]

# ---------------------------------------------------------------------
# 3. Generación de subcolumnas (0-10) con perfiles distintos por
#    empleado, para que existan clusters naturales de desempeño
# ---------------------------------------------------------------------
def generar_subcolumnas(prefijo, nivel_base):
    """Genera valores 0-10 correlacionados a un 'nivel base' por empleado,
    con ruido individual por subcolumna, redondeados a 1 decimal."""
    cols = {}
    for sub in PESOS[prefijo]:
        ruido = rng.normal(0, 1.1, N)
        valores = np.clip(nivel_base + ruido, 0, 10)
        cols[sub] = np.round(valores, 1)
    return cols


def main():
    ids = np.arange(1, N + 1)
    nombres = [f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)} {rng.choice(APELLIDOS)}"
               for _ in range(N)]
    areas = rng.choice(AREAS, N)
    departamentos = [rng.choice(AREAS_DEPTOS[a]) for a in areas]

    hoy = datetime(2026, 9, 7)
    fechas = [hoy - timedelta(days=int(d)) for d in rng.integers(0, 365, N)]

    # Nivel base de desempeño general por empleado (crea 3 "arquetipos"
    # naturales: bajo / medio / alto desempeño en habilidades blandas)
    nivel_general = rng.choice(
        [3.5, 6.0, 8.3], size=N, p=[0.22, 0.48, 0.30]
    ) + rng.normal(0, 0.6, N)

    data = {
        "id_empleado": ids,
        "nombre_completo": nombres,
        "area": areas,
        "departamento": departamentos,
        "fecha_evaluacion": [f.strftime("%Y-%m-%d") for f in fechas],
    }

    for prefijo in ["com", "col", "lid", "res"]:
        # ligera variación de nivel por habilidad para que no sean idénticas
        nivel_habilidad = nivel_general + rng.normal(0, 0.7, N)
        subcols = generar_subcolumnas(prefijo, nivel_habilidad)
        data.update(subcols)
        score = np.zeros(N)
        for sub, peso in PESOS[prefijo].items():
            score += subcols[sub] * peso
        data[f"{prefijo}_score"] = np.round(score, 2)

    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"datos/dataset_habilidades_blandas_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Dataset generado: {len(df)} registros -> {filename}")
    print(df.head())


if __name__ == "__main__":
    main()
