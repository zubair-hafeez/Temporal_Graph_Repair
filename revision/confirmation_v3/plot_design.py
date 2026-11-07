"""Readable overview of the implemented training and repair interfaces."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(6.6, 4.7))
ax.set(xlim=(0, 6.6), ylim=(0, 4.7)); ax.axis('off')
ax.text(1.55, 4.52, 'TRAIN: hold supervision fixed', ha='center', weight='bold', fontsize=10)
ax.text(5.05, 4.52, 'CONFIRM: use new histories', ha='center', weight='bold', fontsize=10)


def box(x, y, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), 2.9, 1.0, boxstyle='round,pad=0.04',
                              facecolor=color, edgecolor='#60747b'))
    ax.text(x + 1.45, y + .73, title, ha='center', va='center', weight='bold', fontsize=9)
    ax.text(x + 1.45, y + .32, body, ha='center', va='center', fontsize=8.5, linespacing=1.4)


def arrow(a, b, **kwargs):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'color': '#4e5d66', 'lw': 1.2, **kwargs})


box(.1, 3.2, 'Budget verified histories', 'Split fitting and calibration;\nreuse eight corrupted fitting views.', '#e7f0f1')
box(.1, 1.85, 'Compare two training arms', 'Real histories alone or with\n32 generated workflow worlds.', '#f4ede6')
box(.1, .5, 'Freeze models and thresholds', '36 existing fits; no tuning on\nnew confirmation histories.', '#e6edf5')
box(3.6, 3.2, 'Read untouched target histories', 'Eight disjoint blocks per source;\nthree paired fault realizations.', '#eff2f4')
box(3.6, 1.85, 'Apply one edit batch', 'Guarded, unguarded or never merge;\natomic root replacements.', '#e6edf5')
box(3.6, .5, 'Measure recovery and harm', 'Audit edits, identities and histories;\npaired block uncertainty.', '#e7f0f1')
for x in [1.55, 5.05]:
    arrow((x, 3.13), (x, 2.91)); arrow((x, 1.78), (x, 1.56))
arrow((3.07, 1.05), (3.53, 2.35), connectionstyle='arc3,rad=-.1')
ax.text(3.27, 1.65, 'fixed models', rotation=72, ha='center', va='center', fontsize=8)
ax.text(3.3, .1, 'Reference identities supervise training; they are unavailable during inference.',
        ha='center', fontsize=8)
fig.tight_layout(pad=.3)
out = ROOT / 'revision/confirmation_v3/figures'
out.mkdir(exist_ok=True)
for extension in ['pdf', 'png']:
    fig.savefig(out / ('study_design.' + extension), dpi=200, bbox_inches='tight')
