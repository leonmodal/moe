from pathlib import Path

from src.utils.routing_plots import plot_routing_snapshot


def test_plot_routing_snapshot_writes_global_and_per_layer_histograms(tmp_path: Path):
    snapshot = {
        "step": 50,
        "layers": {
            0: {"token_counts": [8, 4, 0, 0], "token_fracs": [0.6667, 0.3333, 0.0, 0.0]},
            1: {"token_counts": [1, 3, 4, 0], "token_fracs": [0.125, 0.375, 0.5, 0.0]},
        },
        "global_pool": {"token_counts": [9, 7, 4, 0], "token_fracs": [0.45, 0.35, 0.2, 0.0]},
        "attention": {
            "attn": {
                "layers": {
                    0: {"token_counts": [6, 2, 0, 0], "token_fracs": [0.75, 0.25, 0.0, 0.0]},
                    1: {"token_counts": [2, 2, 2, 2], "token_fracs": [0.25, 0.25, 0.25, 0.25]},
                },
                "global_pool": {
                    "token_counts": [8, 4, 2, 2],
                    "token_fracs": [0.5, 0.25, 0.125, 0.125],
                },
            }
        },
        "norms": {},
    }

    plot_routing_snapshot(snapshot, str(tmp_path), step=50)

    assert (tmp_path / "mlp" / "expert_histogram.png").exists()
    assert (tmp_path / "mlp" / "per_layer_expert_histograms.png").exists()
    assert (tmp_path / "attn" / "expert_histogram.png").exists()
    assert (tmp_path / "attn" / "per_layer_expert_histograms.png").exists()


def test_plot_routing_snapshot_writes_per_layer_histograms_without_global_pool(tmp_path: Path):
    snapshot = {
        "step": 100,
        "layers": {
            "0": {"token_counts": [2, 2], "token_fracs": [0.5, 0.5]},
            "1": {"token_counts": [4, 0], "token_fracs": [1.0, 0.0]},
        },
        "attention": {
            "q": {
                "layers": {
                    "0": {"token_counts": [1, 3], "token_fracs": [0.25, 0.75]},
                },
                "global_pool": None,
            }
        },
        "norms": {},
    }

    plot_routing_snapshot(snapshot, str(tmp_path), step=100)

    assert (tmp_path / "mlp" / "per_layer_expert_histograms.png").exists()
    assert not (tmp_path / "mlp" / "expert_histogram.png").exists()
    assert (tmp_path / "q" / "per_layer_expert_histograms.png").exists()
    assert not (tmp_path / "q" / "expert_histogram.png").exists()
