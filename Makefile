.PHONY: install run_api run_streamlit run_docker

install_dep:
	pip install -r requirements.txt

run_api:
	uvicorn api.fast:app --reload --host 0.0.0.0 --port 8000

run_streamlit:
	streamlit run streamlit/app.py

run_docker:
	docker compose up -d
