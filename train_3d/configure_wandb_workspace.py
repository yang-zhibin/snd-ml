#!/usr/bin/env python3
"""Create the curated W&B workspace used by clustered-GravNet benchmarks."""

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws


ENTITY = "1246249540"
PROJECT = "snd-ml-data-driven-3d"
WORKSPACE_NAME = "Clustered GravNet Benchmark"
WORKSPACE_URL = (
    "https://wandb.ai/1246249540/snd-ml-data-driven-3d?nw=psg5r7h4zch"
)


def line(title, metrics, *, x="Step", y_title=None):
    return wr.LinePlot(
        title=title,
        x=x,
        y=metrics,
        title_x="epoch" if x == "epoch" else "optimizer step",
        title_y=y_title,
        smoothing_type="none",
        legend_position="south",
    )


def main():
    validation_f1 = [
        f"val/f1/{name}"
        for name in (
            "CC_numu",
            "CC_nue",
            "NC",
            "neutral_kaon",
            "neutron",
            "passing_muon",
            "muon_DIS",
        )
    ]
    desired = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=WORKSPACE_NAME,
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(
            x_axis="Step",
            smoothing_type="none",
            max_runs=20,
        ),
        runset_settings=ws.RunsetSettings(
            filters="Config('model_version') = 'clustered_pyg_gravnet_v1'",
            pinned_runs=["u1hvw5st", "tmffdwym"],
            pinned_columns=[
                "run:displayName",
                "run:state",
                "summary:train/loss_epoch",
                "summary:train/accuracy",
                "summary:val/loss",
                "summary:val/accuracy",
                "summary:throughput/events_s",
            ],
        ),
        sections=[
            ws.Section(
                name="Performance curves by step",
                is_open=True,
                panels=[
                    line("Training loss per step", ["train/loss_step"], y_title="loss"),
                    line(
                        "Batch accuracy per step",
                        ["train/accuracy_step"],
                        y_title="accuracy",
                    ),
                    line(
                        "Batch balanced accuracy per step",
                        ["train/balanced_accuracy_step"],
                        y_title="balanced accuracy",
                    ),
                    line(
                        "Batch F1 per step",
                        ["train/macro_f1_step", "train/weighted_f1_step"],
                        y_title="F1",
                    ),
                ],
            ),
            ws.Section(
                name="Performance curves by epoch",
                is_open=True,
                panels=[
                    line(
                        "Training and validation loss by epoch",
                        ["train/loss_epoch", "val/loss"],
                        x="epoch",
                        y_title="loss",
                    ),
                    line(
                        "Training and validation accuracy by epoch",
                        ["train/accuracy", "val/accuracy"],
                        x="epoch",
                        y_title="score",
                    ),
                    line(
                        "Balanced accuracy by epoch",
                        ["train/balanced_accuracy", "val/balanced_accuracy"],
                        x="epoch",
                        y_title="score",
                    ),
                    line(
                        "Macro-F1 by epoch",
                        ["train/macro_f1", "val/macro_f1"],
                        x="epoch",
                        y_title="F1",
                    ),
                    line(
                        "Weighted-F1 by epoch",
                        ["train/weighted_f1", "val/weighted_f1"],
                        x="epoch",
                        y_title="F1",
                    ),
                ],
            ),
            ws.Section(
                name="Validation F1 by class",
                is_open=False,
                panels=[
                    line(
                        "Per-class validation F1 by epoch",
                        validation_f1,
                        x="epoch",
                        y_title="F1",
                    )
                ],
            ),
            ws.Section(
                name="Throughput and timing",
                is_open=True,
                panels=[
                    line("Event throughput", ["throughput/events_s"], y_title="events/s"),
                    line("Node throughput", ["throughput/nodes_s"], y_title="nodes/s"),
                    line(
                        "Dataloader and training step time",
                        ["timing/data_s", "timing/step_s"],
                        y_title="seconds",
                    ),
                ],
            ),
            ws.Section(
                name="GPU memory",
                is_open=True,
                panels=[
                    line(
                        "Peak GPU memory",
                        ["memory/gpu_peak_mb", "memory/gpu_peak_reserved_mb"],
                        y_title="MiB",
                    )
                ],
            ),
            ws.Section(
                name="Batch diagnostics",
                is_open=False,
                panels=[
                    line("Events and nodes per batch", ["batch/events", "batch/nodes"]),
                    line(
                        "Clusters per batch",
                        ["batch/clusters", "batch/max_clusters_per_event"],
                    ),
                    line(
                        "Cluster sizes",
                        ["batch/mean_cluster_nodes", "batch/max_cluster_nodes"],
                        y_title="nodes",
                    ),
                ],
            ),
        ],
    )
    # Update the known view rather than creating duplicate workspaces when this
    # configuration is applied again.
    workspace = ws.Workspace.from_url(WORKSPACE_URL)
    workspace.name = desired.name
    workspace.settings = desired.settings
    workspace.runset_settings = desired.runset_settings
    workspace.sections = desired.sections
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
