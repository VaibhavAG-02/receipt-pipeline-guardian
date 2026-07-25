.PHONY: install pipeline dbt test lint app all all-spark clean stream-up stream-down spark verify-ci

install:
	pip install -r requirements-dev.txt

pipeline:
	PYTHONPATH=src python -m rpg.pipeline --receipts 20000

dbt:
	cd dbt && RPG_WAREHOUSE=../data/warehouse.duckdb dbt build --profiles-dir .

test:
	PYTHONPATH=src python -m pytest tests -q

lint:
	ruff check src tests app

app:
	PYTHONPATH=src streamlit run app/streamlit_app.py

# Executes every `run:` step of .github/workflows/ci.yml locally, in order.
# Catches broken CI before you push. Does not emulate the Actions runner.
verify-ci:
	python scripts/verify_ci.py

all: pipeline dbt test

# Optional Redpanda path
stream-up:
	docker compose up -d
	@echo "Redpanda console: http://localhost:8080"

stream-down:
	docker compose down -v

# Optional Spark path (needs a JDK)
spark:
	PYTHONPATH=src python -m rpg.spark_job --input data/raw --emit-bronze

# Full local run WITH the Spark stage and the cross-engine check.
all-spark: pipeline spark
	cd dbt && RPG_WAREHOUSE=../data/warehouse.duckdb dbt build --profiles-dir . \
		--vars '{spark_bronze: true, spark_bronze_path: ../data/raw/spark_store_health.parquet}'
	PYTHONPATH=src python -m pytest tests -q

clean:
	rm -rf data dbt/target dbt/logs .pytest_cache .ruff_cache
