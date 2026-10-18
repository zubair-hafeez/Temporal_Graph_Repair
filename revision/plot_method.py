"""Readable overview of the implemented training and repair interfaces."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(6.6, 4.7))
ax.set(xlim=(0, 6.6), ylim=(0, 4.7)); ax.axis('off')
ax.text(1.55, 4.52, 'TRAIN: learn edit scores', ha='center', weight='bold', fontsize=10)
ax.text(5.05, 4.52, 'APPLY: restore histories', ha='center', weight='bold', fontsize=10)


def box(x, y, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), 2.9, 1.0, boxstyle='round,pad=0.04',
                              facecolor=color, edgecolor='#60747b'))
    ax.text(x + 1.45, y + .73, title, ha='center', va='center', weight='bold', fontsize=9)
    ax.text(x + 1.45, y + .32, body, ha='center', va='center', fontsize=8.5, linespacing=1.4)


def arrow(a, b, **kwargs):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'color': '#4e5d66', 'lw': 1.2, **kwargs})


box(.1, 3.2, 'Execute workflows', 'Orders, resources and timing;\nretries and cancellations.', '#e7f0f1')
box(.1, 1.85, 'Create observed training graphs', 'Vary fields; inject missing/wrong\nlinks and fragmented identities.', '#f4ede6')
box(.1, .5, 'Fit and calibrate scorers', 'Generated reference labels plus\nbudgeted verified histories.', '#e6edf5')
box(3.6, 3.2, 'Read the observed graph', 'Recorded times, activities, fields\nand typed object handles.', '#eff2f4')
box(3.6, 1.85, 'Propose and apply edits', 'Retrieve candidates; add, remove,\nmerge or abstain; refresh context.', '#e6edf5')
box(3.6, .5, 'Return histories and edit audit', 'Preserve recorded event content.\nUnresolved links may remain.', '#e7f0f1')
for x in [1.55, 5.05]:
    arrow((x, 3.13), (x, 2.91)); arrow((x, 1.78), (x, 1.56))
arrow((3.07, 1.05), (3.53, 2.35), connectionstyle='arc3,rad=-.1')
ax.text(3.27, 1.65, 'fitted model', rotation=72, ha='center', va='center', fontsize=8)
ax.text(3.3, .1, 'Reference identities supervise training; they are unavailable during inference.',
        ha='center', fontsize=8)
fig.tight_layout(pad=.3)
out = ROOT / 'revision/paper/figures'
out.mkdir(exist_ok=True)
for extension in ['pdf', 'png']:
    fig.savefig(out / ('method_overview.' + extension), dpi=200, bbox_inches='tight')
