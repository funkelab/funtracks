test:
    uv run pytest --cov --cov-report=html tests/

docs-serve:
    uv run mike serve

docs-deploy:
    uv run mike deploy
