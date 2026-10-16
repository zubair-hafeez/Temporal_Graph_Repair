"""Vector overview of the implemented training and application boundary."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set(xlim=(0, 13), ylim=(0, 6.4)); ax.axis('off')


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=.08',
                               facecolor=color, edgecolor='#60717a', linewidth=1.1))
    ax.text(x+w/2, y+h-.28, title, ha='center', va='center', weight='bold')
    ax.text(x+w/2, y+.4, body, ha='center', va='center', fontsize=10, linespacing=1.5)


def arrow(a, b, **kwargs):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
                                linewidth=1.4, color='#4e5d66', **kwargs))


ax.text(.12, 6.1, 'TRAINING: generated executions provide repair supervision', weight='bold', fontsize=13)
box(.12, 4.7, 2.55, 1.05, 'Execute workflows', 'Orders, resources, queues,\nretries and exceptions', '#e8f2f2')
box(3.1, 4.7, 2.65, 1.05, 'Vary observations', 'Change descriptive fields;\nretain the same execution', '#e8f2f2')
box(6.2, 4.7, 2.65, 1.05, 'Introduce recording faults', 'Missing links, wrong joins,\nfragmented identifiers', '#f7eee8')
box(9.35, 4.7, 3.4, 1.05, 'Learn edit scores', 'Observed candidates as inputs;\nreference edits as labels', '#e8edf6')
for a, b in [(2.73, 3.02), (5.81, 6.12), (8.91, 9.27)]: arrow((a, 5.22), (b, 5.22))
ax.plot([1.4, 1.4, 10.95], [4.62, 4.08, 4.08], color='#8668a4', linewidth=1.3)
arrow((10.95, 4.08), (10.95, 4.6))
ax.text(6.25, 4.12, 'Executed reference: labels only', ha='center', va='bottom', fontsize=10, color='#72548e')
ax.plot([0, 13], [3.67, 3.67], color='#a5afb5', linestyle='--', linewidth=1)
ax.text(.12, 3.3, 'APPLICATION: repair an available event–object history', weight='bold', fontsize=13)
box(.12, 1.83, 2.55, 1.1, 'Observed records', 'Times, activities, attributes,\ntyped object handles', '#f2f3f4')
box(3.1, 1.83, 2.65, 1.1, 'Retrieve candidates', 'Use timing, activity context\nand observed relations', '#e8f2f2')
box(6.2, 1.83, 2.65, 1.1, 'Score and apply edits', 'Add / remove participation;\nmerge aliases or abstain', '#e8edf6')
box(9.35, 1.83, 3.4, 1.1, 'Retrieve a history', 'Relevant records and exceptions;\naudit of changes', '#e8f2f2')
for a, b in [(2.73, 3.02), (5.81, 6.12), (8.91, 9.27)]: arrow((a, 2.38), (b, 2.38))
ax.plot([7.5, 7.5, 4.42], [1.75, 1.05, 1.05], color='#4e5d66', linewidth=1.3)
arrow((4.42, 1.05), (4.42, 1.75))
ax.text(5.95, .79, 'Refresh context after accepted edits', ha='center', fontsize=10)
arrow((11.92, 4.6), (8.55, 3.0), connectionstyle='arc3,rad=.05')
ax.text(10.55, 3.34, 'Fitted scorer', ha='center', fontsize=10,
        bbox={'facecolor':'white', 'edgecolor':'none', 'pad':2})
ax.text(.12, .16, 'Protected: recorded times, activities and descriptive fields.  Unresolved associations may remain.',
        fontsize=10, color='#4e5d66')
fig.subplots_adjust(left=.015, right=.99, bottom=.03, top=.98)
for suffix in ['pdf', 'png']:
    fig.savefig(ROOT/'paper/figures'/('method_overview.'+suffix), dpi=180, bbox_inches='tight')
