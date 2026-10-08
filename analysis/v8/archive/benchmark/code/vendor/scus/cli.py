from __future__ import annotations

from pathlib import Path
import json
import click

from .config import load_config


def _configured(config_path, input_path=None, output_dir=None, seed=None, device=None):
    cfg = load_config(config_path)
    if input_path is not None:
        key = "input_dir" if cfg.get("workflow") == "pretrain" else "input"
        cfg.setdefault("data", {})[key] = str(Path(input_path).expanduser().resolve())
    if output_dir is not None:
        cfg["output_dir"] = str(Path(output_dir).expanduser().resolve())
    if seed is not None:
        cfg["seed"] = seed
    if device is not None:
        cfg["device"] = device
        if cfg.get("workflow") == "pretrain":
            options = cfg.setdefault("pretrain", {})
            if device == "cpu":
                options.update({"accelerator": "cpu", "devices": 1})
            elif device.startswith("cuda"):
                index = int(device.split(":", 1)[1]) if ":" in device else 0
                options.update({"accelerator": "gpu", "devices": [index]})
    return cfg


def runtime_options(function):
    options = [
        click.option("--input", "input_path", type=click.Path(exists=True)),
        click.option("--output-dir", type=click.Path(file_okay=False)),
        click.option("--seed", type=int),
        click.option("--device"),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="scUS")
def main():
    """scUS pretraining, zero-shot inference, and frozen-encoder alignment."""


@main.group("data")
def data_group(): """Prepare H5AD metadata or pretraining token shards."""


@data_group.command("prepare")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tag")
@runtime_options
def data_prepare(config_path, tag, input_path, output_dir, seed, device):
    cfg = _configured(config_path, input_path, output_dir, seed, device)
    if cfg.get("workflow") == "pretrain":
        from .data.pretraining import prepare_pretraining_shards
        result = prepare_pretraining_shards(cfg)
    else:
        from .data.prepare import prepare_data
        result = prepare_data(cfg, tag)
    click.echo(result)


@data_group.command("merge")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--stage", required=True, type=click.Choice(["zero-shot", "align-project"]))
@click.option("--tag", required=True)
@click.option("--parts", required=True, help="Comma-separated input tags in any order.")
def data_merge(config_path, stage, tag, parts):
    from .zero_shot import merge_parts
    click.echo(merge_parts(load_config(config_path), stage, tag, [value.strip() for value in parts.split(",") if value.strip()]))


@main.command("pretrain")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--resume", type=click.Path(exists=True, dir_okay=False))
@runtime_options
def pretrain_command(config_path, resume, input_path, output_dir, seed, device):
    from .pretrain import train
    click.echo(train(_configured(config_path, input_path, output_dir, seed, device), resume))


@main.command("zero-shot")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tag")
@click.option("--prepare-tag")
@click.option("--start", default=0, type=int)
@click.option("--stop", type=int)
@runtime_options
def zero_shot_command(config_path, tag, prepare_tag, start, stop, input_path, output_dir, seed, device):
    from .zero_shot import encode
    click.echo(encode(_configured(config_path, input_path, output_dir, seed, device), tag, start, stop, device, prepare_tag=prepare_tag))


@main.group("align")
def align_group(): """Fit and apply scUS-Align with a frozen encoder."""


@align_group.command("fit")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tag")
@click.option("--epochs", type=int)
@click.option("--steps-per-epoch", type=int)
@runtime_options
def align_fit(config_path, tag, epochs, steps_per_epoch, input_path, output_dir, seed, device):
    from .align import fit
    cfg = _configured(config_path, input_path, output_dir, seed, device)
    click.echo(fit(cfg, tag, device, epochs, steps_per_epoch))


@align_group.command("project")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tag")
@click.option("--fit-tag")
@click.option("--prepare-tag")
@click.option("--start", default=0, type=int)
@click.option("--stop", type=int)
@runtime_options
def align_project(config_path, tag, fit_tag, prepare_tag, start, stop, input_path, output_dir, seed, device):
    from .align import project
    cfg = _configured(config_path, input_path, output_dir, seed, device)
    click.echo(project(cfg, tag, start, stop, device, fit_tag, prepare_tag))


@main.command("validate")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tag")
@runtime_options
def validate_command(config_path, tag, input_path, output_dir, seed, device):
    from .validation import validate
    cfg = _configured(config_path, input_path, output_dir, seed, device)
    click.echo(json.dumps(validate(cfg, tag), indent=2))


@main.group("diagnostics")
def diagnostics_group():
    """Build auditable optimization, anti-collapse, and downstream diagnostics."""


@diagnostics_group.command("training")
@click.option("--pretrain-log-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--checkpoint-dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--align-log",
    "align_logs",
    multiple=True,
    help="Repeatable PROTOCOL=PATH entry. Missing historical metrics remain NA.",
)
@click.option("--downstream-metrics", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="outputs/paper/training_diagnostics", type=click.Path(file_okay=False))
@click.option("--resume-step", default=191380, show_default=True, type=int)
@click.option("--selected-epoch", default=11, show_default=True, type=int)
@click.option("--seed", default=618, show_default=True, type=int)
def diagnostics_training(pretrain_log_root, checkpoint_dir, align_logs, downstream_metrics, output_dir,
                         resume_step, selected_epoch, seed):
    from .training_diagnostics import run_training_diagnostics

    parsed = []
    for value in align_logs:
        if "=" not in value:
            raise click.BadParameter("Align logs must use PROTOCOL=PATH", param_hint="--align-log")
        protocol, path = value.split("=", 1)
        candidate = Path(path).expanduser()
        if not protocol.strip() or not candidate.is_file():
            raise click.BadParameter(f"Invalid Align log: {value}", param_hint="--align-log")
        parsed.append((protocol.strip(), candidate.resolve()))
    result = run_training_diagnostics(
        pretrain_log_root=Path(pretrain_log_root).resolve(),
        checkpoint_dir=Path(checkpoint_dir).resolve() if checkpoint_dir else None,
        align_logs=parsed,
        downstream_metrics=Path(downstream_metrics).resolve() if downstream_metrics else None,
        output_dir=Path(output_dir).expanduser().resolve(),
        resume_step=resume_step,
        selected_epoch=selected_epoch,
        seed=seed,
    )
    click.echo(result)


@diagnostics_group.command("pretraining-panel-a")
@click.option("--metrics", "metrics_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--val-log-root", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output", default="outputs/paper/training_diagnostics/01A_pretraining_step_loss_styled",
    type=click.Path(dir_okay=False), show_default=True,
)
@click.option("--resume-step", default=191380, show_default=True, type=int)
@click.option("--show-resume/--hide-resume", default=False, show_default=True)
@click.option("--center-window", default=301, show_default=True, type=int)
@click.option("--band-window", default=801, show_default=True, type=int)
def diagnostics_pretraining_panel_a(metrics_path, val_log_root, output, resume_step, show_resume, center_window, band_window):
    """Redraw pretraining panel 1A with rolling curves and variability bands."""
    import pandas as pd
    from .training_diagnostics import plot_pretraining_step_loss_styled, reconstructed_validation_steps

    validation = reconstructed_validation_steps(Path(val_log_root).resolve()) if val_log_root else None
    result = plot_pretraining_step_loss_styled(
        pd.read_csv(Path(metrics_path).resolve(), low_memory=False),
        Path(output).expanduser().resolve(), resume_step if show_resume else None,
        center_window, band_window, validation,
    )
    click.echo(result)


@main.group("ablation")
def ablation_group():
    """Run frozen-checkpoint representation and context ablations."""


@ablation_group.group("frozen")
def frozen_ablation_group():
    """Smoke-test frozen scUS ablations without retraining the Transformer."""


@frozen_ablation_group.command("smoke-cpu")
@click.option("--output-dir", default="outputs/paper/pretraining_ablations/smoke/cpu", type=click.Path(file_okay=False))
@click.option("--seed", default=42, show_default=True, type=int)
def frozen_smoke_cpu(output_dir, seed):
    from .ablations import run_cpu_smoke
    click.echo(run_cpu_smoke(Path(output_dir).expanduser().resolve(), seed))


@frozen_ablation_group.command("smoke-gpu")
@click.option("--cache-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--vocab", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="outputs/paper/pretraining_ablations/smoke/gpu", type=click.Path(file_okay=False))
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--cells", default=256, show_default=True, type=int)
@click.option("--genes", "genes_per_cell", default=128, show_default=True, type=int)
@click.option("--targets", "target_genes", default=8, show_default=True, type=int)
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--preflight-samples", default=12, show_default=True, type=int)
@click.option("--preflight-interval", default=10.0, show_default=True, type=float)
@click.option("--minimum-free-mb", default=40_000.0, show_default=True, type=float)
@click.option("--maximum-mean-utilization", default=20.0, show_default=True, type=float)
@click.option("--checkpoint-label", default="epoch11", show_default=True)
@click.option("--include-random/--no-random", default=True, show_default=True)
@click.option("--run-kind", type=click.Choice(["smoke", "formal"]), default="smoke", show_default=True)
def frozen_smoke_gpu(cache_dir, checkpoint, vocab, output_dir, device, cells, genes_per_cell,
                     target_genes, batch_size, seed, preflight_samples, preflight_interval,
                     minimum_free_mb, maximum_mean_utilization, checkpoint_label,
                     include_random, run_kind):
    from .ablations import run_gpu_smoke
    if device == "cuda:1":
        raise click.ClickException("GPU 1 is excluded by the registered resource policy")
    click.echo(run_gpu_smoke(
        Path(cache_dir).resolve(), Path(checkpoint).resolve(), Path(vocab).resolve(),
        Path(output_dir).expanduser().resolve(), device, cells, genes_per_cell,
        target_genes, batch_size, seed, preflight_samples, preflight_interval,
        minimum_free_mb, maximum_mean_utilization, checkpoint_label,
        include_random, run_kind,
    ))


@frozen_ablation_group.command("cache-formal")
@click.option("--cache-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default="outputs/paper/pretraining_ablations/forebrain_cache", type=click.Path(file_okay=False))
@click.option("--pairs-per-cell", default=512, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--metadata-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--dataset", default="Forebrain", show_default=True)
@click.option("--label-file", default="cell_types.npy", show_default=True)
@click.option("--task", default="cell_type", show_default=True)
@click.option("--sample-side", type=click.Choice(["prefix", "suffix"]), default="prefix", show_default=True)
@click.option("--drop-label", "drop_labels", multiple=True, default=("nan", "unknown", ""), show_default=True)
def frozen_cache_formal(cache_dir, output_dir, pairs_per_cell, seed, metadata_dir, dataset,
                        label_file, task, sample_side, drop_labels):
    from .ablations import run_cache_formal
    click.echo(run_cache_formal(
        Path(cache_dir).resolve(), Path(output_dir).expanduser().resolve(), pairs_per_cell, seed,
        Path(metadata_dir).resolve() if metadata_dir else None, dataset, label_file, task, sample_side,
        tuple(drop_labels),
    ))


@frozen_ablation_group.command("summarize-gpu")
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default="outputs/paper/pretraining_ablations", type=click.Path(file_okay=False))
def frozen_summarize_gpu(input_dir, output_dir):
    """Combine completed checkpoint, context, and top-k GPU ablations."""
    from .ablations import summarize_gpu_formal
    click.echo(summarize_gpu_formal(
        Path(input_dir).resolve(), Path(output_dir).expanduser().resolve()
    ))


@main.group("artifacts")
def artifacts_group(): """Download and verify external model assets."""


@main.group("semantics")
def semantics_group():
    """Falsifiable analysis of frozen scUS U/S distance semantics."""


def _semantics_config(config_path, output_dir, seed, device):
    return _configured(config_path, None, output_dir, seed, device)


@semantics_group.command("audit")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
@click.option("--seed", type=int)
@click.option("--device")
def semantics_audit(config_path, output_dir, seed, device):
    from .semantics_workflow import audit
    click.echo(audit(_semantics_config(config_path, output_dir, seed, device)))


@semantics_group.command("smoke")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
@click.option("--seed", type=int)
def semantics_smoke(config_path, output_dir, seed):
    from .semantics_workflow import cpu_smoke
    click.echo(cpu_smoke(_semantics_config(config_path, output_dir, seed, "cpu")))


def _semantic_stage_options(function):
    options = [
        click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False)),
        click.option("--smoke/--formal", default=True, show_default=True),
        click.option("--output-dir", type=click.Path(file_okay=False)),
        click.option("--seed", type=int), click.option("--device"),
    ]
    for option in reversed(options): function = option(function)
    return function


@semantics_group.command("layerwise")
@_semantic_stage_options
def semantics_layerwise(config_path, smoke, output_dir, seed, device):
    from .semantics_workflow import layerwise
    click.echo(layerwise(_semantics_config(config_path, output_dir, seed, device), smoke, device))


@semantics_group.command("interventions")
@_semantic_stage_options
def semantics_interventions(config_path, smoke, output_dir, seed, device):
    from .semantics_workflow import interventions
    click.echo(interventions(_semantics_config(config_path, output_dir, seed, device), smoke, device))


@semantics_group.command("surprisal")
@_semantic_stage_options
def semantics_surprisal(config_path, smoke, output_dir, seed, device):
    from .semantics_workflow import surprisal
    click.echo(surprisal(_semantics_config(config_path, output_dir, seed, device), smoke, device))


@semantics_group.command("surprisal-empirical")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--smoke/--formal", default=True, show_default=True)
@click.option("--output-dir", type=click.Path(file_okay=False))
@click.option("--seed", type=int)
def semantics_surprisal_empirical(config_path, smoke, output_dir, seed):
    from .semantics_workflow import surprisal_empirical
    click.echo(surprisal_empirical(_semantics_config(config_path, output_dir, seed, "cpu"), smoke))


def _feasibility_command(command_name, stage_name=None):
    stage_name = stage_name or command_name.replace("-", "_")
    def command(config_path, output_dir):
        from .semantics_workflow import feasibility
        click.echo(feasibility(load_config(config_path) | ({"output_dir": str(Path(output_dir).resolve())} if output_dir else {}), stage_name))
    command.__name__ = f"semantics_{stage_name}"
    command = click.option("--output-dir", type=click.Path(file_okay=False))(command)
    command = click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))(command)
    return semantics_group.command(command_name)(command)


@semantics_group.command("temporal")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
def semantics_temporal(config_path, output_dir):
    from .semantics_workflow import temporal
    click.echo(temporal(_semantics_config(config_path, output_dir, None, "cpu")))


@semantics_group.command("simulate")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
@click.option("--smoke/--formal", default=True, show_default=True)
@click.option("--device")
def semantics_simulate(config_path, output_dir, smoke, device):
    from .semantics_workflow import simulate
    click.echo(simulate(_semantics_config(config_path, output_dir, None, device or "cpu"), smoke, device))


@semantics_group.command("reference")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
@click.option("--smoke/--formal", default=True, show_default=True)
def semantics_reference(config_path, output_dir, smoke):
    from .semantics_workflow import reference_anomaly
    click.echo(reference_anomaly(_semantics_config(config_path, output_dir, None, "cpu"), smoke))


@semantics_group.command("align-retention")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
def semantics_align_retention(config_path, output_dir):
    from .semantics_workflow import align_retention
    click.echo(align_retention(_semantics_config(config_path, output_dir, None, "cpu")))


@semantics_group.command("robustness")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
def semantics_robustness(config_path, output_dir):
    from .semantics_workflow import robustness
    click.echo(robustness(_semantics_config(config_path, output_dir, None, "cpu")))


@semantics_group.command("validate")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
def semantics_validate(config_path, output_dir):
    from .semantics_workflow import validate
    click.echo(json.dumps(validate(_semantics_config(config_path, output_dir, None, None)), indent=2))


@semantics_group.command("visualize")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False))
def semantics_visualize(config_path, output_dir):
    from .semantics_workflow import visualize
    click.echo(visualize(_semantics_config(config_path, output_dir, None, None)))


def _iclr_semantics_options(function):
    function = click.option("--output-dir", type=click.Path(file_okay=False))(function)
    function = click.option("--seed", type=int)(function)
    function = click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))(function)
    return function


@semantics_group.command("calibrate")
@_iclr_semantics_options
def semantics_calibrate(config_path, seed, output_dir):
    """Fit gene/bin-conditional distance calibration on reference samples."""
    from .iclr_distance import calibrate
    click.echo(calibrate(_semantics_config(config_path, output_dir, seed, "cpu")))


@semantics_group.command("attribution")
@click.option("--device")
@_iclr_semantics_options
def semantics_attribution(config_path, seed, output_dir, device):
    """Aggregate layer/checkpoint/random-init distance attribution."""
    from .iclr_distance import attribution
    click.echo(attribution(_semantics_config(config_path, output_dir, seed, device)))


@semantics_group.command("benchmark")
@_iclr_semantics_options
def semantics_benchmark(config_path, seed, output_dir):
    """Run held-out profile and incremental distance benchmarks."""
    from .iclr_distance import benchmark
    click.echo(benchmark(_semantics_config(config_path, output_dir, seed, "cpu")))


@semantics_group.command("align-pareto")
@click.option("--device")
@_iclr_semantics_options
def semantics_align_pareto(config_path, seed, output_dir, device):
    """Audit task-versus-relational-semantics trade-offs across Align epochs."""
    from .iclr_distance import align_pareto
    click.echo(align_pareto(_semantics_config(config_path, output_dir, seed, device)))


@semantics_group.command("core-validate")
@_iclr_semantics_options
def semantics_core_validate(config_path, seed, output_dir):
    """Validate the registered ICLR distance-core artifacts and queues."""
    from .iclr_distance import validate_iclr_core
    result = validate_iclr_core(_semantics_config(config_path, output_dir, seed, "cpu"))
    click.echo(json.dumps(result, indent=2))


@artifacts_group.command("download")
@click.argument("name")
@click.option("--output-dir", default="artifacts/downloads", type=click.Path(file_okay=False))
@click.option("--url")
def artifact_download(name, output_dir, url):
    from .artifacts import download
    click.echo(download(name, output_dir, url))


@artifacts_group.command("verify")
@click.argument("name")
@click.option("--path", "artifact_path", required=True, type=click.Path(exists=True, dir_okay=False))
def artifact_verify(name, artifact_path):
    from .artifacts import verify
    if not verify(name, artifact_path): raise click.ClickException("SHA-256 mismatch")
    click.echo("OK")


@main.group("example")
def example_group(): """Dataset-specific reproducible examples."""


@example_group.command("navigo")
@click.argument("action", type=click.Choice(["prepare", "zero-shot", "align", "analyze", "visualize"]))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tag")
@runtime_options
def navigo_command(action, config_path, tag, input_path, output_dir, seed, device):
    from .examples import run_navigo
    cfg = _configured(config_path, input_path, output_dir, seed, device)
    click.echo(run_navigo(action, cfg, tag, device))


if __name__ == "__main__": main()
