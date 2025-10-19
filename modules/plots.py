from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tempfile

def save_profile_plot(md, tvd, x=None, title="Profile"):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(tvd, md, linewidth=2)  # ось Y вниз/вверх — оставим «как есть» для стабильности
    ax.set_xlabel("TVD, м")
    ax.set_ylabel("MD, м")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.tight_layout()
    fig.savefig(tmp.name, dpi=150)
    plt.close(fig)
    return tmp.name

def save_plan_plot(n, e, title="Plan View"):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(e, n, linewidth=2)
    ax.set_xlabel("Easting, м")
    ax.set_ylabel("Northing, м")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.tight_layout()
    fig.savefig(tmp.name, dpi=150)
    plt.close(fig)
    return tmp.name
