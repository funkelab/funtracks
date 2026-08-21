test:
    uv run pytest --cov --cov-report=html tests/ tests_old/

docs-serve:
    uv run mkdocs serve

docs-deploy:
    uv run mike deploy
