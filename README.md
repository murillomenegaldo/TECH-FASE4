# Tech Challenge — Fase 4: LSTM Stock Price Predictor

Modelo preditivo de redes neurais **LSTM** para prever o preço de fechamento da **Petrobras (PETR4.SA)** na bolsa de valores brasileira (B3), com deploy via **FastAPI** e monitoramento com **Prometheus + Grafana**.

---

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Executando a API](#executando-a-api)
- [Docker](#docker)
- [Endpoints da API](#endpoints-da-api)
- [Monitoramento](#monitoramento)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

O projeto implementa a pipeline completa de ML:

1. **Coleta** — `yfinance` para download de dados históricos (2018–2024)
2. **Pré-processamento** — normalização MinMax, criação de janelas temporais (60 dias)
3. **Modelo LSTM** — duas camadas LSTM empilhadas com Dropout
4. **Avaliação** — MAE, RMSE, MAPE, R²
5. **Deploy** — API RESTful com FastAPI
6. **Monitoramento** — métricas Prometheus expostas em `/metrics`

---

## Arquitetura

```
Input (60 dias) → LSTM(128, return_sequences=True) → Dropout(0.2)
               → LSTM(64) → Dropout(0.2)
               → Dense(32, relu) → Dense(1)
               → Preço de fechamento previsto
```

---

## Pré-requisitos

- Python 3.11+
- Docker & Docker Compose (para deploy containerizado)

---

## Instalação

```bash
git clone https://github.com/murillomenegaldo/tech-fase4.git
cd tech-fase4
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Treinamento do Modelo

```bash
# Parâmetros padrão: PETR4.SA, 2018-01-01 → 2024-12-31
python train.py

# Personalizando
python train.py --symbol AAPL --start 2019-01-01 --end 2024-12-31 --epochs 150
```

Artefatos salvos em `saved_model/`:
| Arquivo | Descrição |
|---------|-----------|
| `lstm_model.keras` | Modelo treinado |
| `scaler.pkl` | MinMaxScaler ajustado |
| `metadata.json` | Hiperparâmetros e métricas |
| `training_history.png` | Curvas de loss/MAE |
| `predictions_vs_actual.png` | Predições no conjunto de teste |

---

## Executando a API

```bash
# Após treinar o modelo:
make serve
# ou
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a documentação interativa: `http://localhost:8000/docs`

---

## Docker

### Build & Run completo (API + Prometheus + Grafana)

```bash
# 1. Treinar o modelo (necessário antes do build)
python train.py

# 2. Subir todos os serviços
docker-compose up -d --build

# 3. Verificar status
docker-compose ps
```

| Serviço | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### Apenas a API

```bash
docker build -t stock-lstm-api .
docker run -p 8000:8000 -v $(pwd)/saved_model:/app/saved_model stock-lstm-api
```

---

## Endpoints da API

### `GET /health`

Verifica se a API está saudável e se o modelo está carregado.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "symbol": "PETR4.SA",
  "seq_len": 60
}
```

### `POST /predict/data`

Predição a partir de preços históricos fornecidos pelo usuário.

**Request body:**
```json
{
  "prices": [28.5, 29.1, 28.8, ...],  // mínimo 60 preços
  "steps": 5                           // dias futuros a prever (1-30)
}
```

**Response:**
```json
{
  "last_known_price": 38.42,
  "predictions": [38.91, 39.15, 39.03, 38.78, 39.22],
  "steps": 5
}
```

### `GET /predict/ticker/{ticker}?steps=N`

Predição diretamente de um ticker do Yahoo Finance (baixa os dados automaticamente).

```bash
curl "http://localhost:8000/predict/ticker/PETR4.SA?steps=3"
```

**Response:**
```json
{
  "ticker": "PETR4.SA",
  "last_known_price": 38.42,
  "predictions": [38.91, 39.15, 39.03],
  "steps": 3
}
```

### `GET /metrics`

Métricas no formato Prometheus (latência de inferência, contagem de requisições, etc.).

---

## Monitoramento

O serviço expõe automaticamente métricas para o Prometheus:

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `stock_inference_latency_seconds` | Histogram | Tempo de inferência LSTM |
| `stock_prediction_requests_total` | Counter | Total de requisições por endpoint/status |
| `http_requests_total` | Counter | Requisições HTTP (auto-instrumentadas) |
| `http_request_duration_seconds` | Histogram | Latência HTTP (auto-instrumentada) |

---

## Métricas de Avaliação

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MAE** | `mean(|y - ŷ|)` | Erro médio absoluto em BRL |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penaliza erros maiores |
| **MAPE** | `mean(|y - ŷ| / y) × 100` | Erro percentual médio |
| **R²** | `1 - SS_res / SS_tot` | Proporção da variância explicada |

---

## Estrutura do Projeto

```
TECH-FASE4/
├── model/
│   ├── data_collector.py     # Download yfinance
│   ├── preprocessor.py       # Normalização e janelas temporais
│   └── lstm_model.py         # Arquitetura LSTM (Keras)
├── api/
│   ├── main.py               # FastAPI app + rotas
│   ├── schemas.py            # Pydantic schemas
│   └── predictor.py          # Lógica de inferência
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/
├── saved_model/              # Artefatos do modelo (gerados pelo train.py)
├── data/                     # Dados baixados (CSV)
├── train.py                  # Script de treinamento
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── Makefile
```

---

## Testes

```bash
make test
# ou
pytest tests/ -v
```

---

## Tecnologias

| Categoria | Tecnologia |
|-----------|-----------|
| Dados | yfinance, pandas, NumPy |
| ML | TensorFlow/Keras, scikit-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Deploy | Docker, Docker Compose |
| Monitoramento | Prometheus, Grafana |
| Testes | pytest |
