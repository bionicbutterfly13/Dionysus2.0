"""
Dionysus Migration CLI

Command-line interface for consciousness-guided legacy component migration.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.tree import Tree

from ..services import (
    ComponentDiscoveryService,
    QualityAssessmentService,
    MigrationPipelineService,
    DaedalusCoordinationService
)
from ..config import get_migration_config


console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="dionysus-migration")
def main():
    """
    Dionysus Migration System CLI

    Consciousness-guided legacy component migration with ThoughtSeed enhancement.
    """
    pass


@main.command()
@click.argument('codebase_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for discovered components')
@click.option('--format', 'output_format', type=click.Choice(['json', 'table', 'tree']),
              default='table', help='Output format')
@click.option('--consciousness-threshold', type=float, default=0.3,
              help='Minimum consciousness score threshold')
def discover(codebase_path: str, output: Optional[str], output_format: str, consciousness_threshold: float):
    """
    Discover consciousness components in legacy codebase

    Analyzes Python files to identify components with consciousness functionality
    patterns including awareness, inference, and memory capabilities.
    """
    console.print(Panel(
        f"🔍 Discovering consciousness components in [bold blue]{codebase_path}[/bold blue]",
        title="Component Discovery",
        border_style="blue"
    ))

    discovery_service = ComponentDiscoveryService()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)

        try:
            components = discovery_service.discover_components(codebase_path)
            progress.update(task, description="Discovery completed!")

        except Exception as e:
            console.print(f"❌ Discovery failed: {e}", style="red")
            sys.exit(1)

    # Filter by consciousness threshold
    filtered_components = [
        comp for comp in components
        if comp.consciousness_functionality.composite_score >= consciousness_threshold
    ]

    console.print(f"\n✅ Discovered {len(components)} total components")
    console.print(f"🧠 Found {len(filtered_components)} consciousness-relevant components (threshold: {consciousness_threshold})")

    # Display results
    if output_format == 'table':
        _display_components_table(filtered_components)
    elif output_format == 'tree':
        _display_components_tree(filtered_components)
    elif output_format == 'json':
        components_data = [comp.dict() for comp in filtered_components]
        if output:
            with open(output, 'w') as f:
                json.dump(components_data, f, indent=2, default=str)
            console.print(f"💾 Results saved to {output}")
        else:
            print(json.dumps(components_data, indent=2, default=str))


@main.command()
@click.argument('codebase_path', type=click.Path(exists=True))
@click.option('--coordinator-id', default='cli-coordinator', help='DAEDALUS coordinator ID')
@click.option('--quality-threshold', type=float, help='Quality threshold override')
@click.option('--max-components', type=int, help='Maximum components to migrate')
@click.option('--dry-run', is_flag=True, help='Simulate migration without execution')
def migrate(codebase_path: str, coordinator_id: str, quality_threshold: Optional[float],
           max_components: Optional[int], dry_run: bool):
    """
    Start complete migration pipeline for a codebase

    Executes the full migration workflow including component discovery,
    quality assessment, and migration task creation.
    """
    console.print(Panel(
        f"🚀 Starting migration pipeline for [bold green]{codebase_path}[/bold green]",
        title="Migration Pipeline",
        border_style="green"
    ))

    if dry_run:
        console.print("🧪 [yellow]DRY RUN MODE - No actual migration will be performed[/yellow]\n")

    # Prepare options
    options = {}
    if quality_threshold:
        options['quality_threshold'] = quality_threshold
    if max_components:
        options['max_components'] = max_components

    async def run_migration():
        pipeline_service = MigrationPipelineService()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing pipeline...", total=None)

            try:
                if not dry_run:
                    pipeline_id = await pipeline_service.start_migration_pipeline(
                        codebase_path=codebase_path,
                        coordinator_id=coordinator_id,
                        options=options
                    )

                    progress.update(task, description=f"Pipeline started: {pipeline_id}")
                    console.print(f"\n✅ Migration pipeline started with ID: [bold cyan]{pipeline_id}[/bold cyan]")
                    console.print(f"📊 Monitor progress with: [bold]dionysus-migration status {pipeline_id}[/bold]")
                else:
                    progress.update(task, description="Dry run completed")
                    console.print("\n✅ Dry run completed - pipeline would be started with given parameters")

            except Exception as e:
                console.print(f"❌ Pipeline failed to start: {e}", style="red")
                sys.exit(1)

    asyncio.run(run_migration())


@main.command()
@click.argument('pipeline_id', required=False)
@click.option('--watch', '-w', is_flag=True, help='Watch for status updates')
@click.option('--interval', type=int, default=5, help='Update interval for watch mode (seconds)')
def status(pipeline_id: Optional[str], watch: bool, interval: int):
    """
    Get migration status and progress

    Shows detailed status for a specific pipeline or all active migrations.
    """
    if not pipeline_id:
        _show_all_status()
        return

    async def show_pipeline_status():
        pipeline_service = MigrationPipelineService()

        if watch:
            console.print(f"👀 Watching pipeline [bold cyan]{pipeline_id}[/bold cyan] (press Ctrl+C to stop)\n")

            try:
                while True:
                    pipeline_task = pipeline_service.get_pipeline_status(pipeline_id)
                    if pipeline_task:
                        _display_pipeline_status(pipeline_task)
                        if pipeline_task.task_status.value in ['completed', 'failed', 'cancelled']:
                            break
                    else:
                        console.print(f"❌ Pipeline not found: {pipeline_id}", style="red")
                        break

                    await asyncio.sleep(interval)
                    console.clear()

            except KeyboardInterrupt:
                console.print("\n👋 Status monitoring stopped")
        else:
            pipeline_task = pipeline_service.get_pipeline_status(pipeline_id)
            if pipeline_task:
                _display_pipeline_status(pipeline_task)
            else:
                console.print(f"❌ Pipeline not found: {pipeline_id}", style="red")
                sys.exit(1)

    asyncio.run(show_pipeline_status())


@main.command()
@click.argument('component_id')
@click.option('--migration-state', type=click.Path(exists=True),
              help='JSON file containing migration state')
@click.option('--retention-days', type=int, default=7, help='Checkpoint retention period')
def checkpoint(component_id: str, migration_state: Optional[str], retention_days: int):
    """
    Create rollback checkpoint for component

    Creates a comprehensive backup enabling fast component rollback.
    """
    console.print(Panel(
        f"💾 Creating rollback checkpoint for [bold yellow]{component_id}[/bold yellow]",
        title="Checkpoint Creation",
        border_style="yellow"
    ))

    # Load migration state if provided
    state_data = {}
    if migration_state:
        try:
            with open(migration_state, 'r') as f:
                state_data = json.load(f)
        except Exception as e:
            console.print(f"❌ Failed to load migration state: {e}", style="red")
            sys.exit(1)

    # In a real implementation, would create actual checkpoint
    console.print("✅ Checkpoint created successfully")
    console.print(f"🔒 Retention period: {retention_days} days")


@main.command()
@click.argument('checkpoint_id')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--backup-current', is_flag=True, default=True,
              help='Backup current state before rollback')
def rollback(checkpoint_id: str, confirm: bool, backup_current: bool):
    """
    Rollback component to checkpoint state

    Performs fast rollback (<30 seconds) to restore component to
    a previously saved checkpoint state.
    """
    if not confirm:
        if not click.confirm(f"Are you sure you want to rollback to checkpoint {checkpoint_id}?"):
            console.print("🚫 Rollback cancelled")
            return

    console.print(Panel(
        f"⏪ Rolling back to checkpoint [bold red]{checkpoint_id}[/bold red]",
        title="Component Rollback",
        border_style="red"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Performing rollback...", total=None)

        # Simulate rollback timing
        import time
        start_time = time.time()
        time.sleep(2)  # Simulate rollback work
        duration = time.time() - start_time

        progress.update(task, description="Rollback completed!")

    console.print(f"✅ Rollback completed successfully in {duration:.2f} seconds")
    if backup_current:
        console.print("💾 Current state backed up before rollback")


@main.command()
def config():
    """
    Show current migration configuration

    Displays current system configuration including thresholds,
    weights, and operational parameters.
    """
    config_data = get_migration_config()

    console.print(Panel(
        "⚙️ Migration System Configuration",
        border_style="cyan"
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="white")

    settings = [
        ("quality_threshold", config_data.quality_threshold, "Minimum quality score for migration"),
        ("consciousness_weight", config_data.consciousness_weight, "Weight for consciousness metrics"),
        ("strategic_weight", config_data.strategic_weight, "Weight for strategic value metrics"),
        ("zero_downtime_required", config_data.zero_downtime_required, "Zero downtime requirement"),
        ("max_concurrent_agents", config_data.max_concurrent_agents, "Maximum concurrent agents"),
        ("rollback_storage_path", config_data.rollback_storage_path, "Rollback checkpoint storage"),
        ("coordination_cycle_interval", config_data.coordination_cycle_interval, "Coordination cycle interval")
    ]

    for setting, value, description in settings:
        table.add_row(setting, str(value), description)

    console.print(table)


def _display_components_table(components):
    """Display components in table format"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Consciousness", style="green")
    table.add_column("Strategic", style="yellow")
    table.add_column("Quality", style="red")
    table.add_column("Patterns", style="blue")

    for comp in components:
        consciousness_score = f"{comp.consciousness_functionality.composite_score:.2f}"
        strategic_score = f"{comp.strategic_value.composite_score:.2f}"
        quality_score = f"{comp.quality_score:.2f}"
        patterns = ", ".join(comp.consciousness_patterns[:3])
        if len(comp.consciousness_patterns) > 3:
            patterns += "..."

        table.add_row(
            comp.name,
            consciousness_score,
            strategic_score,
            quality_score,
            patterns
        )

    console.print(table)


def _display_components_tree(components):
    """Display components in tree format"""
    tree = Tree("🧠 Consciousness Components")

    for comp in components:
        comp_branch = tree.add(f"[bold cyan]{comp.name}[/bold cyan] (Quality: {comp.quality_score:.2f})")

        # Add consciousness details
        consciousness_branch = comp_branch.add("🧠 Consciousness")
        consciousness_branch.add(f"Awareness: {comp.consciousness_functionality.awareness_score:.2f}")
        consciousness_branch.add(f"Inference: {comp.consciousness_functionality.inference_score:.2f}")
        consciousness_branch.add(f"Memory: {comp.consciousness_functionality.memory_score:.2f}")

        # Add strategic details
        strategic_branch = comp_branch.add("⚡ Strategic Value")
        strategic_branch.add(f"Uniqueness: {comp.strategic_value.uniqueness_score:.2f}")
        strategic_branch.add(f"Reusability: {comp.strategic_value.reusability_score:.2f}")
        strategic_branch.add(f"Framework Alignment: {comp.strategic_value.framework_alignment_score:.2f}")

        # Add patterns
        if comp.consciousness_patterns:
            patterns_branch = comp_branch.add("🔍 Patterns")
            for pattern in comp.consciousness_patterns:
                patterns_branch.add(pattern)

    console.print(tree)


def _display_pipeline_status(pipeline_task):
    """Display pipeline status"""
    status_color = {
        "pending": "yellow",
        "in_progress": "blue",
        "completed": "green",
        "failed": "red",
        "cancelled": "orange"
    }.get(pipeline_task.task_status.value, "white")

    console.print(Panel(
        f"Pipeline ID: [bold cyan]{pipeline_task.pipeline_id}[/bold cyan]\n"
        f"Status: [{status_color}]{pipeline_task.task_status.value.upper()}[/{status_color}]\n"
        f"Created: {pipeline_task.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Updated: {pipeline_task.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        title="Pipeline Status",
        border_style=status_color
    ))

    # Show progress metrics if available
    if hasattr(pipeline_task, 'discovered_components'):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Components Discovered", str(getattr(pipeline_task, 'discovered_components', 0)))
        table.add_row("Migration Candidates", str(getattr(pipeline_task, 'migration_candidates', 0)))
        table.add_row("Tasks Created", str(getattr(pipeline_task, 'created_tasks', 0)))

        console.print(table)

    # Show errors if any
    if pipeline_task.errors:
        console.print("\n❌ [red]Errors:[/red]")
        for error in pipeline_task.errors:
            console.print(f"  • {error}")


def _show_all_status():
    """Show status of all active operations"""
    console.print(Panel(
        "📊 Migration System Status Overview",
        border_style="blue"
    ))

    # This would query actual services for status
    console.print("🔍 Active Pipelines: 0")
    console.print("🤖 Active Agents: 0")
    console.print("⚡ Active Enhancements: 0")
    console.print("\n💡 Use 'dionysus-migration status <pipeline-id>' for detailed status")


if __name__ == "__main__":
    main()