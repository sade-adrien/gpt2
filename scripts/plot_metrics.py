"""
This script plots the training state (train/vall loss, lr, gradient norm) from a logs.json.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

log_file = 'weights/logs.json'
output_image = 'results/metrics_plot.png'

with open(log_file, 'r') as file:
    data = json.load(file)

steps = [entry['step'] for entry in data]
tokens = [entry['tokens'] for entry in data]
train_loss = [entry['train_loss'] for entry in data]
val_loss = [entry['val_loss'] if entry['val_loss'] is not None else float('nan') for entry in data]
val_loss_mask = np.isfinite(val_loss)
hellaswag_acc = [entry['hellaswag_acc'] if entry['hellaswag_acc'] is not None else float('nan') for entry in data]
hellaswag_acc_mask = np.isfinite(hellaswag_acc)
hellaswag_acc_norm = [entry['hellaswag_acc_norm'] if entry['hellaswag_acc_norm'] is not None else float('nan') for entry in data]
hellaswag_acc_norm_mask = np.isfinite(hellaswag_acc_norm)
learning_rate = [entry['learning_rate'] for entry in data]
gradient_norm = [entry['gradient_norm'] for entry in data]

# plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6), sharex=True)
plt.title('Metrics Over Training Steps')


############## first plot: losses = f(tokens)
ax1.set_xlabel('Tokens')
ax1.set_ylabel('Loss')
ax1.plot(tokens, train_loss, 'b-', label='Train Loss', alpha=.6)
ax1.plot(np.array(tokens)[val_loss_mask], np.array(val_loss)[val_loss_mask], 'r-', label='Validation Loss')
ax1.legend(loc='upper right')
ax1.set_ylim(2., 5.)

ax1.set_title('Loss Metrics')
################################################

############## second plot: hellaswag_acc = f(tokens)
ax2.set_xlabel('Tokens')
ax2.set_ylabel('Accuracy')
ax2.plot(np.array(tokens)[hellaswag_acc_mask], np.array(hellaswag_acc)[hellaswag_acc_mask], 'b-', label='accuracy')
ax2.plot(np.array(tokens)[hellaswag_acc_norm_mask], np.array(hellaswag_acc_norm)[hellaswag_acc_norm_mask], 'r-', label='accuracy_norm')

# add gpt2 baseline to second plot
from matplotlib.lines import Line2D

ax2.axhline(y=0.2859, color='b', linestyle='--', linewidth=1) # acc GPT2 small
line1 = Line2D([0], [0], color='b', linestyle='--', linewidth=1)
ax2.axhline(y=0.2955, color='r', linestyle='--', linewidth=1) # acc_norm GPT2 small
line2 = Line2D([0], [0], color='r', linestyle='-.', linewidth=1)
# ax2.axhline(y=0.3842, color='b', linestyle='-.', linewidth=1) # acc GPT2 xl
# line3 = Line2D([0], [0], color='b', linestyle='-.', linewidth=1)
# ax2.axhline(y=0.4893, color='r', linestyle='-.', linewidth=1) # acc_norm GPT2 xl
# line4 = Line2D([0], [0], color='r', linestyle='-.', linewidth=1)

handles, labels = ax2.get_legend_handles_labels()
# handles.extend([line1, line3, line2, line4])
# labels.extend(['GPT2-124M acc', 'GPT2-1.5B acc', 'GPT2-124M acc_norm', 'GPT2-1.5B acc_norm'])
handles.extend([line1, line2])
labels.extend(['GPT2-124M acc', 'GPT2-124M acc_norm'])
ax2.legend(handles, labels, loc='upper left')

ax2.set_title('Hellaswag Accuracy')
################################################


############## third plot: lr and gradient_norm = f(tokens)
ax3.set_xlabel('Tokens')
ax3.set_ylabel('Learning Rate', color='g')
line1, = ax3.plot(tokens, learning_rate, 'g--', label='Learning Rate', markersize=8)
ax3_twin = ax3.twinx()
ax3_twin.set_ylabel('Gradient Norm', color='k')
line2, = ax3_twin.plot(tokens, gradient_norm, 'k--', label='Gradient Norm', alpha=.3)
lines, labels = [], []
lines.extend([line1, line2])
labels.extend(['Learning Rate', 'Gradient Norm'])
ax3.legend(lines, labels, loc='upper right')
ax3.set_title('Learning Rate and Gradient Norm')
ax3_twin.grid(False)  
################################################

plt.tight_layout()

plt.savefig(output_image)