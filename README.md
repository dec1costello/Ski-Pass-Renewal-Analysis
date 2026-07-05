![Status](https://img.shields.io/badge/status-active-success.svg)
![Domain](https://img.shields.io/badge/domain-ML%20%7C%20Analytics-blue.svg)
![Warehouse](https://img.shields.io/badge/warehouse-DuckDB-orange.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![dbt](https://img.shields.io/badge/dbt-1.5+-orange.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-0.9+-yellow.svg)
![ML](https://img.shields.io/badge/LGBMClassifier%20%7C%20Optuna-purple.svg)
<br />
[GitHub](https://github.com/dec1costello) | [Kaggle](https://www.kaggle.com/dec1costello) | [LinkedIn](https://www.linkedin.com/in/declan-costello-7423aa137/)
<br />
Author: Declan Costello

<p align="center">
<img height="150" width="150" src="https://github.com/user-attachments/assets/c4cd95c9-ddd4-43a7-9765-aad6b49ed62d"/>  

</p>

<h1 align="center">Ski Pass Renewal Prediction</h1>


**Business Objective:** Predict ski pass renewal likelihood to maximize lifetime customer value. Prioritized recall for the 'will not renew' class to ensure coverage of the at-risk population.

**Technical Approach:** A production grade ML pipeline leveraging modern analytics engineering patterns. Features are defined in SQL (dbt), persisted in DuckDB, and consumed by Python ML workflows, ensuring reproducibility, version control, and auditability.

**Key Results:**
- ✅ **Feature Store:** SQL defined, version controlled feature engineering
- ✅ **Reproducible ML:** Deterministic training with hyperparameter optimization
- ✅ **Operational Analytics:** Predictions stored as tables for BI consumption
- ✅ **Minimal Infrastructure:** Single DuckDB file serves as both warehouse and feature store

**Architecture Rationale:** DuckDB was selected over Snowflake/AWS for its embedded nature, eliminating cloud costs while maintaining SQL compliance and performance for datasets under 100GB. This ski pass renewal prediction system operates on a "single source of truth" principle with DuckDB as the central analytical engine. Customer data flows through automated cleaning and feature engineering pipelines, then machine learning models generate renewal likelihood scores that are directly stored as business ready tables. This design eliminates data silos and infrastructure complexity, allowing marketing teams to immediately access predictive insights through standard business intelligence tools while data scientists maintain full reproducibility.

<br />

## 🚀 Quick Start

This project uses **[uv](https://docs.astral.sh/uv/)** for deterministic dependency management for reproducible ML and pipelines. The entire environment is defined in `pyproject.toml` and locked in `uv.lock`, guaranteeing identical execution across development, CI, and production.

#### Prerequisites
- **Python 3.11+**
- **Git**
- **Terminal access**

#### 1. Environment Setup

```bash
# Install uv (one time system setup)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or via pip
pip install uv

# Clone and setup the project
git clone https://github.com/dec1costello/Ski-Pass-Renewal-Prediction.git
cd ski-pass-renewal-prediction

# Recreate the exact development environment
uv sync --frozen
```

#### 2. Verify Installation

```bash
# Test critical imports
uv run python -c "import duckdb, xgboost, sklearn; print('✓ Environment ready')"

# Check dbt availability
uv run dbt --version
```

#### 3. Execution Workflow

Execute the complete pipeline using the following commands:

| Step | Purpose | Command |
|------|---------|---------|
| **Data Pipeline** | Transform raw data → silver → gold features | `uv run dbt run --select silver+` |
| **Training** | Train model with hyperparameter optimization | `uv run python src/models/train.py` |
| **Evaluation** | Generate performance metrics and validation | `uv run python src/models/evaluate.py` |
| **Prediction** | Run batch inference on latest data | `uv run python src/models/predict.py` |

> [!TIP]
> Use `uv run` before any Python command to guarantee execution with the locked environment. This ensures consistent Python versions and dependency trees across all machines.








