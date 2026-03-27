"""
Bar chart summary of test accuracy for all 9 models across 3 evaluation conditions.
Reads logs/test_summary.csv and saves to plots/summary_accuracy.png.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./plots', exist_ok=True)

with open('logs/test_summary.csv') as f:
    rows = {r['model']: r for r in csv.DictReader(f)}

MODELS = [
    'Plain-Old-CIFAR10-FC', 'Plain-Old-CIFAR10-CNN', 'Plain-Old-CIFAR10-Attention',
    'D-shuffletruffle-FC',  'D-shuffletruffle-CNN',  'D-shuffletruffle-Attention',
    'N-shuffletruffle-FC',  'N-shuffletruffle-CNN',  'N-shuffletruffle-Attention',
]
LABELS = [m.replace('Plain-Old-CIFAR10', 'CIFAR10').replace('shuffletruffle', 'shuf') for m in MODELS]

test_acc    = [float(rows[m]['test_acc'])    for m in MODELS]
patch16_acc = [float(rows[m]['patch16_acc']) for m in MODELS]
patch8_acc  = [float(rows[m]['patch8_acc'])  for m in MODELS]

x = np.arange(len(MODELS))
w = 0.26

fig, ax = plt.subplots(figsize=(14, 6))
b1 = ax.bar(x - w,   test_acc,    w, label='Test (original)',  color='#1f77b4')
b2 = ax.bar(x,       patch16_acc, w, label='Patch-16 shuffle', color='#ff7f0e')
b3 = ax.bar(x + w,   patch8_acc,  w, label='Patch-8 shuffle',  color='#2ca02c')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Test Accuracy by Model and Evaluation Condition')
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=9)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar in [b1, b2, b3]:
    for rect in bar:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5)

plt.tight_layout()
out = './plots/summary_accuracy.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
