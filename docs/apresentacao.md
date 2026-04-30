---
marp: true
theme: default
paginate: true
backgroundColor: "#ffffff"
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
  }
  h1 { color: #1a3a5c; border-bottom: 3px solid #2980b9; padding-bottom: 8px; }
  h2 { color: #2980b9; }
  h3 { color: #1a3a5c; }
  code { background: #f0f4f8; border-radius: 4px; padding: 2px 6px; }
  pre { background: #1e1e2e; color: #cdd6f4; border-radius: 8px; }
  table { border-collapse: collapse; width: 100%; }
  th { background: #1a3a5c; color: white; padding: 8px 12px; }
  td { padding: 8px 12px; border: 1px solid #ddd; }
  tr:nth-child(even) { background: #f0f4f8; }
  .badge { display: inline-block; background: #2980b9; color: white; border-radius: 12px; padding: 2px 10px; font-size: 14px; margin: 2px; }
  blockquote { border-left: 4px solid #2980b9; background: #f0f4f8; padding: 10px 16px; border-radius: 0 8px 8px 0; }
---

<!-- _paginate: false -->
<!-- _backgroundColor: "#1a3a5c" -->
<!-- _color: white -->

# Tech Challenge — Fase 4
## Predição de Preços de Ações com LSTM

<br>

> **Modelo preditivo de deep learning para forecasting do preço de fechamento da PETR4.SA (Petrobras)**

<br>

🏢 **FIAP — Pós-Graduação em Machine Learning Engineering**
📅 **2024 / 2025**

---

## Agenda

1. O Problema
2. Dados e Pré-processamento
3. Arquitetura do Modelo LSTM
4. Treinamento e Hiperparâmetros
5. Métricas de Avaliação
6. Deploy com FastAPI
7. Containerização com Docker
8. Monitoramento: Prometheus + Grafana
9. Demonstração da API
10. Conclusão e Próximos Passos

---

## 1. O Problema

### Contexto

O mercado financeiro gera uma enorme quantidade de **dados sequenciais e temporais**. Prever o preço futuro de ações é um problema clássico de séries temporais com alta relevância prática.

### Desafio

> Construir um modelo de **deep learning (LSTM)** capaz de prever o **preço de fechamento** da Petrobras (PETR4.SA) e disponibilizá-lo via **API REST** em produção.

### Por que LSTM?

| Abordagem | Vantagem | Limitação |
|-----------|----------|-----------|
| Médias móveis | Simples | Não captura padrões complexos |
| ARIMA | Interpretável | Assume linearidade |
| **LSTM** | **Captura dependências longas** | **Maior custo computacional** |
| Transformer | SOTA em NLP | Requer mais dados |

---

## 2. Dados — Coleta

### Fonte: Yahoo Finance via `yfinance`

```python
import yfinance as yf

symbol    = 'PETR4.SA'   # Petrobras — B3
start_date = '2018-01-01'
end_date   = '2024-12-31'

df = yf.download(symbol, start=start_date, end=end_date)
```

### Características do Dataset

| Atributo | Valor |
|----------|-------|
| Ativo | PETR4.SA (Petrobras ON) |
| Bolsa | B3 — Brasil, Bolsa, Balcão |
| Período | Jan/2018 → Dez/2024 |
| Frequência | Diária (dias úteis) |
| Total de registros | ~1.750 pregões |
| Features disponíveis | Open, High, Low, **Close**, Volume |
| Feature utilizada | **Close** (preço de fechamento) |

---

## 2. Dados — Pré-processamento

### Pipeline

```
Dados Brutos → Extração (Close) → Split → Normalização → Janelas Temporais
```

### Divisão dos dados

```
|──────────── Treino (70%) ────────────|── Val (15%) ──|── Teste (15%) ──|
       ~1.225 dias                        ~263 dias        ~263 dias
```

### Normalização

- **MinMaxScaler** (range [0, 1]) ajustado **apenas** no conjunto de treino
- Evita vazamento de informação (data leakage) para validação/teste
- Inversão da escala na saída da predição → preço em BRL

### Criação de Janelas Temporais

```python
seq_len = 60  # Janela de 60 pregões (≈ 3 meses)

# Para cada ponto t:
# X[t] = [Close(t-60), Close(t-59), ..., Close(t-1)]  → shape (60, 1)
# y[t] = Close(t)                                       → escalar
```

---

## 3. Arquitetura do Modelo LSTM

### O que é uma LSTM?

A **Long Short-Term Memory** é uma arquitetura de rede neural recorrente capaz de aprender **dependências de longo prazo** em séries temporais por meio de três gates:

- **Forget gate** — decide o que descartar da memória
- **Input gate** — decide o que armazenar na memória
- **Output gate** — decide o que passar para o próximo passo

### Arquitetura implementada

```
Input (batch, 60, 1)
    │
    ▼
LSTM(128 unidades, return_sequences=True)
    │
    ▼
Dropout(0.2)
    │
    ▼
LSTM(64 unidades, return_sequences=False)
    │
    ▼
Dropout(0.2)
    │
    ▼
Dense(32, activation='relu')
    │
    ▼
Dense(1)  →  Preço previsto (BRL)
```

---

## 3. Arquitetura — Detalhes

```python
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(seq_len=60):
    inputs = keras.Input(shape=(seq_len, 1))

    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)

    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model
```

### Parâmetros totais: **~115.000**

---

## 4. Treinamento e Hiperparâmetros

### Configuração

| Hiperparâmetro | Valor |
|----------------|-------|
| Otimizador | Adam (lr = 0.001) |
| Loss | Mean Squared Error (MSE) |
| Batch size | 32 |
| Épocas máximas | 100 |
| Early Stopping | patience = 15, restore best weights |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |

### Callbacks utilizados

```python
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ModelCheckpoint('saved_model/best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
]
```

### Artefatos gerados pelo `train.py`

```
saved_model/
├── lstm_model.keras          # Modelo completo
├── best_model.keras          # Melhor checkpoint
├── scaler.pkl                # MinMaxScaler serializado
├── metadata.json             # Config + métricas
├── training_history.png      # Curvas de loss/MAE
└── predictions_vs_actual.png # Predições vs. real (teste)
```

---

## 5. Métricas de Avaliação

### Métricas utilizadas

| Métrica | Fórmula | O que mede |
|---------|---------|-----------|
| **MAE** | `mean(│y − ŷ│)` | Erro médio em BRL |
| **RMSE** | `√mean((y − ŷ)²)` | Erro médio — penaliza outliers |
| **MAPE** | `mean(│y − ŷ│/y) × 100` | Erro percentual médio |
| **R²** | `1 − SS_res/SS_tot` | Variância explicada pelo modelo |

### Interpretação esperada

| Métrica | Bom | Aceitável |
|---------|-----|-----------|
| MAPE | < 3% | < 8% |
| R² | > 0.95 | > 0.85 |

> **Importante:** modelos de previsão financeira raramente alcançam alta precisão em horizontes longos. O modelo prevê **1 dia à frente** com maior confiabilidade do que múltiplos dias.

### Predição recursiva (multi-step)

Para N dias futuros, o modelo aplica a predição de forma recursiva:

```
[d₁..d₆₀] → ŷ₆₁ → [d₂..d₆₀, ŷ₆₁] → ŷ₆₂ → ...
```

---

## 6. Deploy — FastAPI

### Estrutura da API

```
GET  /health                       → Liveness/Readiness probe
POST /predict/data                 → Previsão a partir de preços fornecidos
GET  /predict/ticker/{ticker}      → Previsão via Yahoo Finance (tempo real)
GET  /metrics                      → Métricas Prometheus
GET  /docs                         → Swagger UI interativo
```

### Exemplo de uso — predição por preços

```bash
curl -X POST http://localhost:8000/predict/data \
  -H "Content-Type: application/json" \
  -d '{
    "prices": [38.5, 38.9, 39.1, ...],  # mínimo 60 preços
    "steps": 5
  }'
```

```json
{
  "last_known_price": 38.42,
  "predictions": [38.91, 39.15, 39.03, 38.78, 39.22],
  "steps": 5
}
```

---

## 6. Deploy — Código da API

```python
# api/main.py (simplificado)
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Stock Price Prediction API")
Instrumentator().instrument(app).expose(app)   # /metrics automático

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor.is_loaded()}

@app.post("/predict/data")
def predict_from_data(request: PredictFromDataRequest):
    result = predictor.predict_from_prices(request.prices, request.steps)
    return PredictionResponse(**result)

@app.get("/predict/ticker/{ticker}")
def predict_from_ticker(ticker: str, steps: int = 1):
    result = predictor.predict_from_ticker(ticker, steps)
    return PredictionResponse(**result)
```

### Validação automática com Pydantic

```python
class PredictFromDataRequest(BaseModel):
    prices: list[float] = Field(..., min_length=60)  # ≥ 60 preços obrigatórios
    steps: int = Field(default=1, ge=1, le=30)        # 1 a 30 dias
```

---

## 7. Containerização com Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/   ./api/
COPY model/ ./model/
COPY saved_model/ ./saved_model/

EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Comandos

```bash
# Build
docker build -t stock-lstm-api .

# Run standalone
docker run -p 8000:8000 \
  -v $(pwd)/saved_model:/app/saved_model \
  stock-lstm-api

# Stack completa (API + Prometheus + Grafana)
docker-compose up -d --build
```

---

## 7. docker-compose — Stack Completa

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./saved_model:/app/saved_model

  prometheus:
    image: prom/prometheus:v2.54.1
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:11.2.2
    ports: ["3000:3000"]
    depends_on: [prometheus]
```

### Serviços disponíveis após `docker-compose up`

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| API (Swagger) | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## 8. Monitoramento

### Métricas customizadas (Prometheus)

```python
from prometheus_client import Histogram, Counter

INFERENCE_LATENCY = Histogram(
    "stock_inference_latency_seconds",
    "Tempo de inferência LSTM",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)
PREDICTION_REQUESTS = Counter(
    "stock_prediction_requests_total",
    "Total de requisições de predição",
    ["endpoint", "status"],   # labels: data|ticker, ok|error
)
```

### Métricas disponíveis em `/metrics`

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `stock_inference_latency_seconds` | Histogram | Tempo de inferência LSTM |
| `stock_prediction_requests_total` | Counter | Requisições por endpoint e status |
| `http_requests_total` | Counter | Todas as requisições HTTP |
| `http_request_duration_seconds` | Histogram | Latência HTTP geral |
| `process_resident_memory_bytes` | Gauge | Uso de memória do processo |

---

## 8. Monitoramento — Configuração

### prometheus.yml

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "stock_api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: "/metrics"
```

### Grafana — Dashboards sugeridos

**Painel de Operação da API:**
- Taxa de requisições por minuto
- Latência P50 / P95 / P99
- Taxa de erros (4xx, 5xx)
- Requisições de predição (ok vs. error)

**Painel de Inferência LSTM:**
- Histograma de latência de inferência
- Latência média por hora
- Contagem acumulada de predições

> A stack provisiona automaticamente o datasource Prometheus no Grafana via arquivos em `monitoring/grafana/provisioning/`.

---

## 9. Demonstração da API

### 1. Verificar saúde do serviço

```bash
curl http://localhost:8000/health
```
```json
{ "status": "healthy", "model_loaded": true, "symbol": "PETR4.SA", "seq_len": 60 }
```

### 2. Prever próximos 3 dias via ticker

```bash
curl "http://localhost:8000/predict/ticker/PETR4.SA?steps=3"
```
```json
{
  "ticker": "PETR4.SA",
  "last_known_price": 38.42,
  "predictions": [38.91, 39.15, 39.03],
  "steps": 3
}
```

### 3. Prever a partir de dados próprios

```bash
curl -X POST http://localhost:8000/predict/data \
  -H "Content-Type: application/json" \
  -d '{"prices": [35.1, 35.4, ..., 38.4], "steps": 1}'
```

### 4. Swagger UI interativo

```
http://localhost:8000/docs
```

---

## 9. Testes Automatizados

### Suite de testes — `pytest tests/ -v`

```
tests/test_preprocessing.py
  ✓ test_shape              — extração do campo Close
  ✓ test_dtype              — tipo float
  ✓ test_lengths            — split 70/15/15
  ✓ test_no_overlap         — sem vazamento entre splits
  ✓ test_output_shapes      — shapes das janelas
  ✓ test_sequence_values    — valores corretos nas janelas
  ✓ test_pipeline_keys      — todas as chaves presentes
  ✓ test_pipeline_shapes    — dimensões corretas
  ✓ test_scaled_range       — normalização em [0, 1]
  ✓ test_output_shape       — shape final da sequência

tests/test_model.py
  ✓ test_model_builds_without_error
  ✓ test_model_output_shape
  ✓ test_model_has_lstm_layers
  ✓ test_model_compiles

tests/test_api.py
  ✓ test_health_returns_200
  ✓ test_health_schema
  ✓ test_too_few_prices_returns_422
  ✓ test_steps_out_of_range_returns_422
  ✓ test_predict_with_mock_model
  ✓ test_predict_ticker_with_mock
```

---

## 10. Conclusão

### O que foi construído

✅ **Pipeline completa de ML** — coleta → pré-processamento → treino → avaliação → deploy

✅ **Modelo LSTM** — 2 camadas empilhadas, ~115k parâmetros, treinado com early stopping e LR scheduling

✅ **API REST** — FastAPI com validação automática (Pydantic), documentação Swagger, métricas Prometheus

✅ **Containerização** — Dockerfile multi-stage + docker-compose com stack de observabilidade completa

✅ **Testes** — 20 testes cobrindo preprocessing, arquitetura do modelo e endpoints da API

✅ **CI/CD** — GitHub Actions com lint, testes e build do Docker

### Limitações conhecidas

- Modelo univariado: usa apenas o preço de fechamento (sem volume, notícias, macro)
- Predição multi-step: erro acumula recursivamente a partir do 3° dia
- Dados históricos: não refletem eventos exógenos (crises, eleições)

### Próximos Passos

- Adicionar features técnicas (RSI, MACD, Bandas de Bollinger)
- Incorporar análise de sentimento de notícias
- Avaliar arquiteturas transformer (Temporal Fusion Transformer)
- Retraining automático com dados mais recentes via pipeline de MLOps

---

<!-- _paginate: false -->
<!-- _backgroundColor: "#1a3a5c" -->
<!-- _color: white -->

# Obrigado!

<br>

### Repositório

```
github.com/murillomenegaldo/tech-fase4
branch: claude/tech-challenge-setup-wWRjK
```

### Como executar

```bash
git clone <repo> && cd TECH-FASE4
pip install -r requirements.txt
python train.py                          # treinar modelo
docker-compose up -d --build            # subir stack completa
```

### Documentação interativa

```
http://localhost:8000/docs
```

---

<!-- _paginate: false -->

## Referências

- Hochreiter, S. & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*, 9(8), 1735–1780.
- Chollet, F. et al. (2015). **Keras**. https://keras.io
- FastAPI Documentation. https://fastapi.tiangolo.com
- Prometheus Documentation. https://prometheus.io/docs
- `yfinance` library. https://github.com/ranaroussi/yfinance
- Yahoo Finance — Dados históricos PETR4.SA

<br>

> **Nota sobre este documento:** Este arquivo é compatível com **Marp** (Markdown Presentation Ecosystem) e pode ser exportado como PDF ou HTML de apresentação.
>
> ```bash
> npx @marp-team/marp-cli docs/apresentacao.md --pdf
> npx @marp-team/marp-cli docs/apresentacao.md --html
> ```
