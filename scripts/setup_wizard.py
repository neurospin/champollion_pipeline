#!/usr/bin/env python3
"""Interactive installation wizard for champollion_pipeline.

Ask three questions (location, use case, GPU), then print and optionally
run the correct pixi commands.

Usage:
    pixi run setup
    python scripts/setup_wizard.py [--dry-run]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import NamedTuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
except ImportError:
    print("rich is not installed. Run: pip install rich>=13.0", file=sys.stderr)
    sys.exit(1)

console = Console()

# ── Location ──────────────────────────────────────────────────────────────────

LOCATIONS = {
    "1": "Local workstation",
    "2": "Jean-Zay (SLURM)",
    "3": "Other remote / cluster",
}

# ── Use cases ─────────────────────────────────────────────────────────────────

USE_CASES = {
    "1": "Full pipeline  (Morphologist → cortical tiles → embeddings)",
    "2": "Embeddings inference only  (you already have .arg graphs)",
    "3": "Model training",
    "4": "Documentation / development only",
    "5": "Everything (full pipeline + docs)",
}

# ── Command matrix ─────────────────────────────────────────────────────────────


class Plan(NamedTuple):
    commands: list[str]
    warnings: list[str]
    notes: list[str]


def build_plan(location: str, use_case: str, gpu: bool) -> Plan:
    commands: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    jean_zay = location == "2"

    if jean_zay:
        notes.append(
            "Jean-Zay: always pass -e <env> to pixi. Never use the default environment."
        )
        notes.append("Use srun / sbatch from slurm/ for heavy compute stages.")

    if use_case == "1":  # full pipeline
        if jean_zay:
            commands += [
                "pixi run -e embeddings install-embeddings",
            ]
            notes.append(
                "BrainVISA / Morphologist is pre-installed on Jean-Zay. "
                "Run cortical-tiles and embeddings steps via SLURM scripts in slurm/."
            )
        else:
            commands.append("pixi run install-all")

    elif use_case == "2":  # embeddings inference
        commands.append("pixi run -e embeddings install-embeddings")

    elif use_case == "3":  # training
        commands.append("pixi run -e training install-embeddings")

    elif use_case == "4":  # docs
        commands.append("pixi run -e docs build-docs")

    elif use_case == "5":  # everything
        if jean_zay:
            commands += [
                "pixi run -e embeddings install-embeddings",
                "pixi run -e docs build-docs",
            ]
        else:
            commands += [
                "pixi run install-all",
                "pixi run -e docs build-docs",
            ]

    if not gpu and use_case in {"1", "2", "3", "5"}:
        warnings.append(
            "No GPU detected or GPU unavailable — inference and training will run on CPU and may be slow."
        )

    return Plan(commands=commands, warnings=warnings, notes=notes)


# ── Prompts ────────────────────────────────────────────────────────────────────


def ask_choice(question: str, choices: dict[str, str]) -> str:
    console.print(f"\n[bold]{question}[/bold]")
    for key, label in choices.items():
        console.print(f"  [{key}] {label}")
    valid = set(choices)
    while True:
        answer = Prompt.ask("Choice", choices=list(valid), show_choices=False)
        if answer in valid:
            return answer
        console.print(f"[red]Enter one of: {', '.join(sorted(valid))}[/red]")


def ask_where() -> str:
    return ask_choice("Where are you running this pipeline?", LOCATIONS)


def ask_use_case() -> str:
    return ask_choice("What will you use the pipeline for?", USE_CASES)


def ask_gpu(use_case: str) -> bool:
    if use_case not in {"1", "2", "3", "5"}:
        return False
    return Confirm.ask("\nIs a GPU available in your environment?", default=True)


# ── Display ────────────────────────────────────────────────────────────────────


def show_plan(plan: Plan, location: str, use_case: str) -> None:
    console.print()

    table = Table(title="Installation plan", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Command", style="green")
    for i, cmd in enumerate(plan.commands, 1):
        table.add_row(str(i), cmd)
    console.print(table)

    if plan.warnings:
        for w in plan.warnings:
            console.print(Panel(f"[yellow]Warning:[/yellow] {w}", border_style="yellow"))

    if plan.notes:
        for n in plan.notes:
            console.print(f"[dim]Note:[/dim] {n}")


# ── Execution ──────────────────────────────────────────────────────────────────


def run_commands(commands: list[str]) -> None:
    for cmd in commands:
        console.print(f"\n[bold cyan]$ {cmd}[/bold cyan]")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            console.print(f"[red]Command failed with exit code {result.returncode}[/red]")
            sys.exit(result.returncode)
    console.print("\n[bold green]Installation complete.[/bold green]")


# ── Main ───────────────────────────────────────────────────────────────────────


def main(dry_run: bool = False) -> None:
    if not shutil.which("pixi"):
        console.print(
            "[red]pixi not found.[/red] Install it from [link=https://pixi.sh]https://pixi.sh[/link]",
            highlight=False,
        )
        sys.exit(1)

    console.print(
        Panel(
            "[bold]Champollion Pipeline — Installation Wizard[/bold]\n"
            "Answer three questions to get the right install commands.",
            border_style="cyan",
        )
    )

    location = ask_where()
    use_case = ask_use_case()
    gpu = ask_gpu(use_case)

    plan = build_plan(location, use_case, gpu)

    if not plan.commands:
        console.print("[yellow]No installation commands for this combination.[/yellow]")
        sys.exit(0)

    show_plan(plan, location, use_case)

    if dry_run:
        console.print("\n[dim]--dry-run: commands not executed.[/dim]")
        return

    if Confirm.ask("\nRun these commands now?", default=True):
        run_commands(plan.commands)
    else:
        console.print("[dim]Commands not run. Copy and run them manually.[/dim]")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
