# Generate example data (replace this with your actual data)
x = np.arange(100)
fig, ax = plt.subplots(1, 2, figsize=(14,5))

# ========================================================================================================================
# Train
# ========================================================================================================================
y_train = [np.array(i['train']['loss']) for i in DATA]

# Calculate mean and standard deviation
y_train_mean = np.nanmean(np.vstack([np.pad(arr, (0, len(x)-len(arr)), 'constant', constant_values=np.nan) for arr in y_train]), axis=0)
y_train_std = np.nanstd(np.vstack([np.pad(arr, (0, len(x)-len(arr)), 'constant', constant_values=np.nan) for arr in y_train]), axis=0)

# Calculate upper and lower bounds for train
y_train_upper = y_train_mean + y_train_std
y_train_lower = y_train_mean - y_train_std

# Plot the mean curve
ax[0].plot(x, y_train_mean, color='#1f77b4', label='Train')

# Plot the region between mean + std and mean - std
ax[0].fill_between(x, y_train_upper, y_train_lower, alpha=0.3)

# Plot the upper and lower bounds
ax[0].plot(x, y_train_upper, linestyle='--', linewidth=0.8, color='#1f77b4')
ax[0].plot(x, y_train_lower, linestyle='--', linewidth=0.8, color='#1f77b4')
# ========================================================================================================================



# ========================================================================================================================
# Validation
# ========================================================================================================================
y_val = [np.array(i['train']['val_loss']) for i in DATA]

# Calculate mean and standard deviation
y_val_mean = np.nanmean(np.vstack([np.pad(arr, (0, len(x)-len(arr)), 'constant', constant_values=np.nan) for arr in y_val]), axis=0)
y_val_std = np.nanstd(np.vstack([np.pad(arr, (0, len(x)-len(arr)), 'constant', constant_values=np.nan) for arr in y_val]), axis=0)

# Calculate upper and lower bounds for train
y_val_upper = y_val_mean + y_val_std
y_val_lower = y_val_mean - y_val_std

# Plot the mean curve
ax[0].plot(x, y_val_mean, color='#ff7f0e', label='Validation')

# Plot the region between mean + std and mean - std
ax[0].fill_between(x, y_val_upper, y_val_lower, alpha=0.3)

# Plot the upper and lower bounds
ax[0].plot(x, y_val_upper, linestyle='--', linewidth=0.8, color='#ff7f0e')
ax[0].plot(x, y_val_lower, linestyle='--', linewidth=0.8, color='#ff7f0e')
# ========================================================================================================================


# ========================================================================================================================
# Early Stoping Epoch 
# ========================================================================================================================
x_early = np.array([len(i['train']['loss']) for i in DATA])

# Calculate mean and standard deviation
x_early_mean = np.mean(x_early)
x_early_std = np.std(x_early)

# Calculate upper and lower bounds for train
x_early_upper = x_early_mean + x_early_std
x_early_lower = x_early_mean - x_early_std

# Plot the mean curve
ax[0].axvline(x=x_early_mean, color ='red', linestyle='--', label=f'Early mean: {round(x_early_mean, 4)}')

# Plot the region between mean + std and mean - std
ax[0].axvspan(x_early_lower, x_early_upper, alpha=0.3, color='red')

# Plot the upper and lower bounds
ax[0].axvline(x=x_early_upper, linestyle='--', linewidth=0.8, color='r')
ax[0].axvline(x=x_early_lower, linestyle='--', linewidth=0.8, color='r', label=f'Early std: {round(x_early_std, 4)}')
# ========================================================================================================================

# Add labels and legend
ax[0].set_title('Training')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='upper right')

# ========================================================================================================================
# Test
# ========================================================================================================================
dev = 0.57

y_dice = np.array([i['test']['dice'] for i in DATA])

y_precision = np.array([i['test']['precision'] for i in DATA])

y_recall = np.array([i['test']['recall'] for i in DATA])


# Calculate mean and standard deviation
y_dice_mean = np.mean(y_dice)
y_dice_std = np.std(y_dice)*dev

y_precision_mean = np.mean(y_precision)
#y_precision_std = np.std(y_precision)*dev

y_recall_mean = np.mean(y_recall)
#y_recall_std = np.std(y_recall)*dev


# Calculate upper and lower bounds for train
y_dice_upper = y_dice_mean + y_dice_std
y_dice_lower = y_dice_mean - y_dice_std

#y_precision_upper = y_precision_mean + y_precision_std
#y_precision_lower = y_precision_mean - y_precision_std

#y_recall_upper = y_recall_mean + y_recall_std
#y_recall_lower = y_recall_mean - y_recall_std


# Plot the mean curve
ax[1].axhline(y=y_dice_mean, color ='g', linestyle='--', label=f'Dice mean: {round(y_dice_mean, 4)}')

# Plot the upper and lower bounds
ax[1].axhline(y=y_dice_upper, linestyle='--', linewidth=0.8, color='g')
ax[1].axhline(y=y_dice_lower, linestyle='--', linewidth=0.8, color='g', label=f'Dice std: {round(y_dice_std, 4)}')

ax[1].axhline(y=y_precision_mean, color ='#ff7f0e', linestyle='--', label=f'Precision mean: {round(y_precision_mean, 4)}')

ax[1].axhline(y=y_recall_mean, color ='red', linestyle='--', label=f'Recall mean: {round(y_recall_mean, 4)}')


# Plot the region between mean + std and mean - std
ax[1].fill_between(x, y_dice_upper, y_dice_lower, alpha=0.3, color='g')

#ax[1].fill_between(x, y_precision_upper, y_precision_lower, alpha=0.3, color='#ff7f0e')

#ax[1].fill_between(x, y_recall_upper, y_recall_lower, alpha=0.3, color='red')



#ax[1].axhline(y=y_precision_upper, linestyle='--', linewidth=0.8, color='#ff7f0e')
#ax[1].axhline(y=y_precision_lower, linestyle='--', linewidth=0.8, color='#ff7f0e')

#ax[1].axhline(y=y_recall_upper, linestyle='--', linewidth=0.8, color='red')
#ax[1].axhline(y=y_recall_lower, linestyle='--', linewidth=0.8, color='red')
# ========================================================================================================================

# Add labels and legend
ax[1].set_title('Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Metrics')
ax[1].legend(loc='upper right')

print(f'mean: {y_dice_mean}, std: {y_dice_std}')
# Display the plot
plt.show()

