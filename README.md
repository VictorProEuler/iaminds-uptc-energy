# IAMinds UPTC – Predicción de Energía, Anomalías y Recomendaciones

Solución para el reto IAMinds: pronóstico de consumo energético por sede (UPTC), detección de anomalías (p99 sobre residuales) y recomendaciones operativas accionables. Incluye reportes listos para consumo por agente/chatbot y un dashboard básico.

## Qué hay en este repositorio

- `reports/` (artefactos para demo/agent/dashboard)
  - `metrics_energy.json`: métricas del modelo (MAE/RMSE) y configuración del split temporal.
  - `pred_vs_real_energy.csv`: comparación real vs predicho en periodo de prueba.
  - `anomalies_energy.csv`: anomalías detectadas (umbral p99 sobre |residual|).
  - `recommendations_energy.csv`: acciones recomendadas asociadas a anomalías.
  - `forecast_24h_energy.csv`: pronóstico próximas 24 horas por sede.
  - `forecast_7d_energy.csv`: pronóstico próximos 7 días por sede.
- `models/`
  - `model_energy.joblib`: modelo entrenado para `energia_total_kwh`.
- `app/`
  - `main.py`: dashboard (Streamlit) que **lee** los archivos de `reports/` (no reentrena).
- `requirements.txt`: dependencias mínimas del proyecto.

## Enfoque técnico (resumen)

- **Target:** `energia_total_kwh`
- **Features principales:** variables temporales (`hora`, `dia_semana`, `mes`, `es_fin_semana`), `sede_id` (one-hot) y señales externas disponibles (`temperatura_exterior_c`, `ocupacion_pct`, `es_festivo`).
- **Validación:** split temporal por timestamp (entrenamiento en pasado, prueba en futuro).
- **Anomalías:** residuales del modelo; anomalía = casos en el percentil 99 de `abs_residual`.
- **Recomendaciones:** reglas operativas simples basadas en severidad y contexto (fuera de horario / fin de semana).
- **Forecast futuro:** generación de timestamps futuros por sede y predicción con el modelo entrenado.

## Dashboard

El dashboard está en `app/main.py` y muestra:
- métricas del modelo,
- pronósticos 24h/7d por sede,
- tablas de anomalías y recomendaciones,
- evidencia real vs pred del periodo de prueba.

> Nota: el dashboard no recalcula el modelo; solo visualiza los artefactos en `reports/`.

## Uso de los reportes (para agente/chatbot)

El agente debe responder leyendo `reports/` y evitando inferencias no soportadas por los datos. Archivos clave:
- métricas: `metrics_energy.json`
- pronóstico: `forecast_24h_energy.csv`, `forecast_7d_energy.csv`
- anomalías: `anomalies_energy.csv`
- recomendaciones: `recommendations_energy.csv`
- evidencia: `pred_vs_real_energy.csv`
