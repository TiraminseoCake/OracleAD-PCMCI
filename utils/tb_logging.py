"""TensorBoard writer selection + score-histogram / score-curve helpers.

Exposes:
    SummaryWriter  -- selected class (torch.utils.tensorboard preferred, then tensorboardX)
    TB_BACKEND     -- string label for the backend that was loaded, or None
    plt            -- matplotlib.pyplot if available, else None
    tb_log_score_histograms(writer, prefix, step, labels, score_t_dict, start_idx)
    tb_log_score_curves(writer, prefix, step, labels, score_t_dict, max_points=2000)
"""
import numpy as np


SummaryWriter = None
TB_BACKEND = None
try:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
    SummaryWriter = _TorchSummaryWriter
    TB_BACKEND = "torch.utils.tensorboard"
except Exception:
    try:
        from tensorboardX import SummaryWriter as _XSummaryWriter
        SummaryWriter = _XSummaryWriter
        TB_BACKEND = "tensorboardX"
    except Exception:
        SummaryWriter = None
        TB_BACKEND = None

plt = None
try:
    import matplotlib.pyplot as _plt
    plt = _plt
except Exception:
    plt = None


def tb_log_score_histograms(writer, prefix, step, labels, score_t_dict, start_idx):
    if writer is None:
        return
    valid = np.isfinite(score_t_dict["A_t"][start_idx:])
    yv = labels[start_idx:][valid]
    for key in ["P_t", "C_t", "G_t", "S_t", "A_t"]:
        sv = score_t_dict[key][start_idx:][valid]
        if len(sv) > 0:
            writer.add_histogram(f"{prefix}/scores/{key}_all", sv, step)
        if (yv == 1).sum() > 0:
            writer.add_histogram(f"{prefix}/scores/{key}_anom", sv[yv == 1], step)
        if (yv == 0).sum() > 0:
            writer.add_histogram(f"{prefix}/scores/{key}_norm", sv[yv == 0], step)


def tb_log_score_curves(writer, prefix, step, labels, score_t_dict, max_points=2000):
    if writer is None or plt is None:
        return
    T = len(labels)
    idx = np.arange(T)
    sel = np.linspace(0, T - 1, max_points).astype(int) if T > max_points else idx

    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(sel, labels[sel], linewidth=1.0)
    axes[0].set_ylabel("label")
    for ax, key in zip(axes[1:], ["P_t", "C_t", "G_t", "S_t", "A_t"]):
        ax.plot(sel, score_t_dict[key][sel], linewidth=1.0)
        ax.set_ylabel(key[:-2])
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    try:
        writer.add_figure(f"{prefix}/figures/score_curves", fig, global_step=step)
    except Exception as e:
        print(f"[warn] tb figure logging failed for {prefix} step {step}: {e}", flush=True)
    finally:
        plt.close(fig)
