from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer
import uvicorn

from mtci.config import ConfigError, load_config
from mtci.execution import run_profile
from mtci.server import create_app

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config: str = typer.Option("mtci.yml", "--config"),
    profile: str = typer.Option("pr-fast", "--profile"),
    out: str = typer.Option("mtci_artifacts", "--out"),
):
    """Run metamorphic testing under a profile."""
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2)

    exit_code, out_dir = run_profile(cfg, profile, out)
    typer.echo(f"Artifacts: {out_dir}")
    raise typer.Exit(code=exit_code)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start a local FastAPI inference server."""
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


@app.command()
def doctor(config: str = typer.Option("mtci.yml", "--config")):
    """Validate config and endpoint connectivity."""
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if cfg.model.mode == "endpoint":
        base_url = cfg.model.base_url.rstrip("/")
        health_url = f"{base_url}/health"
        predict_url = f"{base_url}{cfg.model.predict_path}"
        try:
            with httpx.Client(timeout=cfg.model.timeout_s) as client:
                health = client.get(health_url)
                health.raise_for_status()
                response = client.post(predict_url, json={"inputs": ["hello"]})
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            typer.secho(f"Endpoint check failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=2)
        if "scores" not in data:
            typer.secho("Endpoint response missing 'scores'", fg=typer.colors.RED)
            raise typer.Exit(code=2)
        typer.echo("Endpoint connectivity: ok")

    typer.echo("Config validation: ok")
