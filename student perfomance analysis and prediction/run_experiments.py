import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
BASE_DIR = r"C:\Users\kspra\Desktop\student perfomance analysis and prediction"
DATA_PATH = os.path.join(BASE_DIR, "student-mat.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

print("--- 1. LOADING DATASET ---")
df = pd.read_csv(DATA_PATH, sep=';')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Summary of target G3
g3_stats = {
    "count": int(df['G3'].count()),
    "mean": float(df['G3'].mean()),
    "std": float(df['G3'].std()),
    "median": float(df['G3'].median()),
    "mode": float(df['G3'].mode()[0]),
    "min": float(df['G3'].min()),
    "q25": float(df['G3'].quantile(0.25)),
    "q75": float(df['G3'].quantile(0.75)),
    "iqr": float(df['G3'].quantile(0.75) - df['G3'].quantile(0.25)),
    "max": float(df['G3'].max()),
    "skewness": float(df['G3'].skew()),
    "kurtosis": float(df['G3'].kurt()),
    "zero_count": int((df['G3'] == 0).sum()),
    "zero_percentage": float((df['G3'] == 0).mean() * 100)
}
print("G3 Statistics:", json.dumps(g3_stats, indent=2))

num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
            'G1', 'G2', 'G3']

num_summary = {}
for col in num_cols:
    num_summary[col] = {
        "mean": round(float(df[col].mean()), 3),
        "std": round(float(df[col].std()), 3),
        "median": round(float(df[col].median()), 3),
        "mode": round(float(df[col].mode()[0]), 3),
        "min": round(float(df[col].min()), 3),
        "max": round(float(df[col].max()), 3),
        "iqr": round(float(df[col].quantile(0.75) - df[col].quantile(0.25)), 3),
        "skewness": round(float(df[col].skew()), 3)
    }

print("--- 2. GENERATING EDA VISUALIZATIONS ---")

# FIGURE 1: Final Grade Distribution (Histogram + KDE + Boxplot)
fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [0.2, 0.8]})
sns.boxplot(x=df['G3'], ax=ax_box, color='#4A90E2', fliersize=4)
ax_box.set(xlabel='')
ax_box.set_title('Figure 1: Distribution Analysis of Student Final Grades (G3)', fontsize=14, fontweight='bold', pad=12)

sns.histplot(df['G3'], kde=True, ax=ax_hist, color='#2E5B88', bins=21, stat="count", discrete=True)
ax_hist.axvline(df['G3'].mean(), color='#D9534F', linestyle='--', linewidth=2, label=f"Mean = {df['G3'].mean():.2f}")
ax_hist.axvline(df['G3'].median(), color='#5CB85C', linestyle='-', linewidth=2, label=f"Median = {df['G3'].median():.2f}")
ax_hist.set_xlabel('Final Grade G3 (0 - 20 Scale)', fontsize=12)
ax_hist.set_ylabel('Student Count', fontsize=12)
ax_hist.set_xticks(range(0, 21))
ax_hist.legend(loc='upper right', frameon=True)

# Annotate zero spike
ax_hist.annotate(f'Zero Grade Spike\n(N = {g3_stats["zero_count"]}, {g3_stats["zero_percentage"]:.1f}%)',
                 xy=(0, g3_stats["zero_count"]), xytext=(2, 45),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="#FFEEBA"))
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_g3_distribution.png'), dpi=300)
plt.close()
print("Saved fig1_g3_distribution.png")

# FIGURE 2: Study Time vs Final Grades (Boxplot with Jittered Strip Plot)
fig, ax = plt.subplots(figsize=(9, 6))
study_labels = ['< 2 hours', '2 to 5 hours', '5 to 10 hours', '> 10 hours']
sns.boxplot(x='studytime', y='G3', data=df, hue='studytime', palette='Blues', ax=ax, width=0.5, boxprops=dict(alpha=0.8), legend=False)
sns.stripplot(x='studytime', y='G3', data=df, color='#E74C3C', alpha=0.4, jitter=0.2, size=5, ax=ax)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(study_labels, fontsize=11)
ax.set_title('Figure 2: Final Grade (G3) Variation Across Weekly Study Time Levels', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Weekly Study Time Category', fontsize=12)
ax.set_ylabel('Final Grade G3 (0 - 20)', fontsize=12)

# Compute mean per study time
means = df.groupby('studytime')['G3'].mean().values
for i, m in enumerate(means):
    ax.text(i, m + 0.6, f"Mean: {m:.2f}", horizontalalignment='center', size=10, weight='bold', color='#1B4F72')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_studytime_vs_grades_boxplot.png'), dpi=300)
plt.close()
print("Saved fig2_studytime_vs_grades_boxplot.png")

# FIGURE 3: Count Plot of Past Failures and Educational Support
fig, ax = plt.subplots(figsize=(9, 6))
sns.countplot(x='failures', hue='schoolsup', data=df, palette={'yes': '#E67E22', 'no': '#3498DB'}, ax=ax)
ax.set_title('Figure 3: Frequency of Past Class Failures Segmented by Extra Educational Support', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Number of Past Class Failures', fontsize=12)
ax.set_ylabel('Number of Students', fontsize=12)
ax.legend(title='Extra School Support', labels=['No Support', 'Extra Support'])
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                    textcoords='offset points')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_failures_studytime_countplot.png'), dpi=300)
plt.close()
print("Saved fig3_failures_studytime_countplot.png")

# FIGURE 4: Correlation Heatmap
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True,
            linewidths=0.5, annot_kws={"size": 9}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Figure 4: Pearson Correlation Matrix Across All Numerical Features', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_correlation_heatmap.png'), dpi=300)
plt.close()
print("Saved fig4_correlation_heatmap.png")

# FIGURE 5: Absences vs Final Grades Scatter Plot
fig, ax = plt.subplots(figsize=(9, 6))
sns.regplot(x='absences', y='G3', data=df,
            scatter_kws={'alpha': 0.6, 'color': '#2980B9', 's': 35},
            line_kws={'color': '#C0392B', 'linewidth': 2}, ax=ax)
r_val, p_val = stats.pearsonr(df['absences'], df['G3'])
ax.set_title(f'Figure 5: Bivariate Relationship Between School Absences and Final Grade (G3)\n(Pearson r = {r_val:.3f}, p-value = {p_val:.4f})',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Number of School Absences (Days)', fontsize=12)
ax.set_ylabel('Final Grade G3 (0 - 20)', fontsize=12)

extreme = df[df['absences'] > 40]
for idx, row in extreme.iterrows():
    ax.annotate(f"Absences: {int(row['absences'])}, G3: {int(row['G3'])}",
                xy=(row['absences'], row['G3']),
                xytext=(row['absences'] - 12, row['G3'] + 2.5),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig5_absences_vs_g3_scatter.png'), dpi=300)
plt.close()
print("Saved fig5_absences_vs_g3_scatter.png")

# FIGURE 6: Grade Progression Trajectory (G1 & G2 vs G3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
sns.regplot(x='G1', y='G3', data=df, ax=ax1,
            scatter_kws={'alpha': 0.5, 'color': '#8E44AD'}, line_kws={'color': '#2C3E50'})
r1, p1 = stats.pearsonr(df['G1'], df['G3'])
ax1.set_title(f'Period 1 Grade (G1) vs Final Grade (G3)\nr = {r1:.3f}', fontsize=12, fontweight='bold')
ax1.set_xlabel('Period 1 Grade G1', fontsize=11)
ax1.set_ylabel('Final Grade G3', fontsize=11)

sns.regplot(x='G2', y='G3', data=df, ax=ax2,
            scatter_kws={'alpha': 0.5, 'color': '#16A085'}, line_kws={'color': '#D35400'})
r2, p2 = stats.pearsonr(df['G2'], df['G3'])
ax2.set_title(f'Period 2 Grade (G2) vs Final Grade (G3)\nr = {r2:.3f}', fontsize=12, fontweight='bold')
ax2.set_xlabel('Period 2 Grade G2', fontsize=11)
ax2.set_ylabel('Final Grade G3', fontsize=11)
fig.suptitle('Figure 6: Academic Momentum and Longitudinal Progression Across Examination Periods', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig6_period_grades_trajectory.png'), dpi=300)
plt.close()
print("Saved fig6_period_grades_trajectory.png")

# FIGURE 7: Alcohol Consumption and Parental Education
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
sns.boxplot(x='Walc', y='G3', data=df, hue='Walc', palette='Purples', ax=ax1, legend=False)
ax1.set_title('Weekend Alcohol Consumption (Walc) vs G3', fontsize=12, fontweight='bold')
ax1.set_xlabel('Weekend Alcohol Consumption (1 = Very Low to 5 = Very High)', fontsize=11)
ax1.set_ylabel('Final Grade G3', fontsize=11)

edu_matrix = df.pivot_table(index='Medu', columns='Fedu', values='G3', aggfunc='mean')
sns.heatmap(edu_matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'Mean G3'})
ax2.set_title('Mean Final Grade by Mother & Father Education Level', fontsize=12, fontweight='bold')
ax2.set_xlabel("Father's Education Level (0 = None to 4 = Higher Ed)", fontsize=11)
ax2.set_ylabel("Mother's Education Level (0 = None to 4 = Higher Ed)", fontsize=11)
fig.suptitle('Figure 7: Socio-Behavioral Analysis: Alcohol Consumption and Parental Education Impact', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig7_alcohol_family_impact.png'), dpi=300)
plt.close()
print("Saved fig7_alcohol_family_impact.png")

print("--- 3. DATA PREPROCESSING & TRAIN-VAL-TEST SPLIT ---")

# Target
y = df['G3'].values.astype(float)

# Setting A: Without G1 and G2 (Early Warning / Pre-exam Prediction, avoiding target leakage)
X_no_grades = df.drop(columns=['G1', 'G2', 'G3'])

# Setting B: With G1 and G2 (Cumulative Performance Prediction)
X_with_grades = df.drop(columns=['G3'])

cat_cols_A = X_no_grades.select_dtypes(include=['object']).columns.tolist()
num_cols_A = X_no_grades.select_dtypes(include=['int64', 'float64']).columns.tolist()

cat_cols_B = X_with_grades.select_dtypes(include=['object']).columns.tolist()
num_cols_B = X_with_grades.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Split Setting B & A using identical indices (70% train, 15% validation, 15% test)
indices = np.arange(len(df))
train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED)

print(f"Dataset Partitioning: Train = {len(train_idx)}, Validation = {len(val_idx)}, Test = {len(test_idx)}")

# Scale target strictly on training data
y_mean = y[train_idx].mean()
y_std = y[train_idx].std()

y_train_sc = (y[train_idx] - y_mean) / y_std
y_val_sc = (y[val_idx] - y_mean) / y_std
y_test_raw = y[test_idx]

# Preprocessor for Setting A
preprocessor_A = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols_A),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols_A)
    ]
)

X_train_A = preprocessor_A.fit_transform(X_no_grades.iloc[train_idx])
X_val_A = preprocessor_A.transform(X_no_grades.iloc[val_idx])
X_test_A = preprocessor_A.transform(X_no_grades.iloc[test_idx])

# Preprocessor for Setting B
preprocessor_B = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols_B),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols_B)
    ]
)

X_train_B = preprocessor_B.fit_transform(X_with_grades.iloc[train_idx])
X_val_B = preprocessor_B.transform(X_with_grades.iloc[val_idx])
X_test_B = preprocessor_B.transform(X_with_grades.iloc[test_idx])

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y_test_raw

ohe_feature_names = preprocessor_B.named_transformers_['cat'].get_feature_names_out(cat_cols_B)
all_feature_names_B = num_cols_B + list(ohe_feature_names)
print(f"Total features after One-Hot Encoding: {len(all_feature_names_B)}")

print("--- 4. BASELINE MODELS ---")
# 1. Dummy Mean Baseline
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train_B, y_train)
y_pred_dummy = dummy.predict(X_test_B)

mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
mse_dummy = mean_squared_error(y_test, y_pred_dummy)
rmse_dummy = np.sqrt(mse_dummy)
r2_dummy = r2_score(y_test, y_pred_dummy)

# 2. Linear Regression Baseline (Setting B - With G1, G2)
lin_reg_B = LinearRegression()
lin_reg_B.fit(X_train_B, y_train)
y_pred_lin_B = lin_reg_B.predict(X_test_B)

mae_lin_B = mean_absolute_error(y_test, y_pred_lin_B)
mse_lin_B = mean_squared_error(y_test, y_pred_lin_B)
rmse_lin_B = np.sqrt(mse_lin_B)
r2_lin_B = r2_score(y_test, y_pred_lin_B)

# 3. Linear Regression Baseline (Setting A - Without G1, G2)
lin_reg_A = LinearRegression()
lin_reg_A.fit(X_train_A, y_train)
y_pred_lin_A = lin_reg_A.predict(X_test_A)

mae_lin_A = mean_absolute_error(y_test, y_pred_lin_A)
mse_lin_A = mean_squared_error(y_test, y_pred_lin_A)
rmse_lin_A = np.sqrt(mse_lin_A)
r2_lin_A = r2_score(y_test, y_pred_lin_A)

print(f"Dummy Baseline: MAE = {mae_dummy:.3f}, MSE = {mse_dummy:.3f}, RMSE = {rmse_dummy:.3f}, R2 = {r2_dummy:.3f}")
print(f"Linear Reg (Setting B - With G1/G2): MAE = {mae_lin_B:.3f}, MSE = {mse_lin_B:.3f}, RMSE = {rmse_lin_B:.3f}, R2 = {r2_lin_B:.3f}")
print(f"Linear Reg (Setting A - No G1/G2): MAE = {mae_lin_A:.3f}, MSE = {mse_lin_A:.3f}, RMSE = {rmse_lin_A:.3f}, R2 = {r2_lin_A:.3f}")

print("--- 5. BUILDING & TRAINING DEEP LEARNING MLP ---")

def build_regularized_mlp(input_dim):
    tf.random.set_seed(SEED)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.015)),
        layers.Dropout(0.2, seed=SEED),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.015)),
        layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss='mse',
        metrics=['mae']
    )
    return model

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

# Train MLP for Setting B (Primary Model for final evaluation with G1, G2)
mlp_B = build_regularized_mlp(X_train_B.shape[1])
mlp_B.summary()

history_B = mlp_B.fit(
    X_train_B, y_train_sc,
    validation_data=(X_val_B, y_val_sc),
    epochs=120,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
)

# Predict Setting B
y_pred_mlp_B_sc = mlp_B.predict(X_test_B, verbose=0).flatten()
y_pred_mlp_B = y_pred_mlp_B_sc * y_std + y_mean

mae_mlp_B = mean_absolute_error(y_test, y_pred_mlp_B)
mse_mlp_B = mean_squared_error(y_test, y_pred_mlp_B)
rmse_mlp_B = np.sqrt(mse_mlp_B)
r2_mlp_B = r2_score(y_test, y_pred_mlp_B)

print(f"\n--- MLP SETTING B (With G1, G2) TEST RESULTS ---")
print(f"MAE:  {mae_mlp_B:.3f}")
print(f"MSE:  {mse_mlp_B:.3f}")
print(f"RMSE: {rmse_mlp_B:.3f}")
print(f"R²:   {r2_mlp_B:.3f}")

# Train MLP for Setting A (Without G1, G2 - Early warning demonstration)
mlp_A = build_regularized_mlp(X_train_A.shape[1])
history_A = mlp_A.fit(
    X_train_A, y_train_sc,
    validation_data=(X_val_A, y_val_sc),
    epochs=120,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
)

y_pred_mlp_A_sc = mlp_A.predict(X_test_A, verbose=0).flatten()
y_pred_mlp_A = y_pred_mlp_A_sc * y_std + y_mean

mae_mlp_A = mean_absolute_error(y_test, y_pred_mlp_A)
mse_mlp_A = mean_squared_error(y_test, y_pred_mlp_A)
rmse_mlp_A = np.sqrt(mse_mlp_A)
r2_mlp_A = r2_score(y_test, y_pred_mlp_A)

print(f"\n--- MLP SETTING A (Without G1, G2) TEST RESULTS ---")
print(f"MAE:  {mae_mlp_A:.3f}")
print(f"MSE:  {mse_mlp_A:.3f}")
print(f"RMSE: {rmse_mlp_A:.3f}")
print(f"R²:   {r2_mlp_A:.3f}")

print("--- 6. MODEL EVALUATION VISUALIZATIONS ---")

# Convert loss back to original scale (MSE on unscaled target) for intuitive interpretation
train_loss_original_scale = [l * (y_std**2) for l in history_B.history['loss']]
val_loss_original_scale = [l * (y_std**2) for l in history_B.history['val_loss']]

# FIGURE 8: Training and Validation Loss Curves
fig, ax = plt.subplots(figsize=(9, 5.5))
epochs_range = range(1, len(history_B.history['loss']) + 1)
ax.plot(epochs_range, train_loss_original_scale, label='Training Loss (MSE)', color='#2980B9', linewidth=2.2)
ax.plot(epochs_range, val_loss_original_scale, label='Validation Loss (MSE)', color='#E74C3C', linewidth=2.2, linestyle='--')
best_epoch = np.argmin(val_loss_original_scale) + 1
best_val_loss = min(val_loss_original_scale)
ax.scatter(best_epoch, best_val_loss, color='#27AE60', s=80, zorder=5, label=f'Best Val Loss ({best_val_loss:.2f} at epoch {best_epoch})')

ax.set_title('Figure 8: Deep Learning MLP Training and Validation Loss vs Epochs (Setting B)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Loss (Mean Squared Error)', fontsize=12)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig8_training_validation_loss.png'), dpi=300)
plt.close()
print("Saved fig8_training_validation_loss.png")

# FIGURE 9: Actual vs Predicted Final Grades
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred_mlp_B, color='#34495E', edgecolors='#2980B9', s=65, alpha=0.85, label='Student Predictions')
# Perfect prediction line
min_val = min(min(y_test), min(y_pred_mlp_B)) - 1
max_val = max(max(y_test), max(y_pred_mlp_B)) + 1
ax.plot([min_val, max_val], [min_val, max_val], color='#C0392B', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')

ax.set_title(f'Figure 9: Actual vs Predicted Final Grades (Test Set)\n(R² = {r2_mlp_B:.3f}, MAE = {mae_mlp_B:.2f}, RMSE = {rmse_mlp_B:.2f})',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Actual Final Grade G3', fontsize=12)
ax.set_ylabel('Predicted Final Grade G3', fontsize=12)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig9_actual_vs_predicted.png'), dpi=300)
plt.close()
print("Saved fig9_actual_vs_predicted.png")

# FIGURE 10: Residual Error Distribution
residuals = y_test - y_pred_mlp_B
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.histplot(residuals, kde=True, color='#16A085', bins=15, ax=ax, stat='density')
ax.axvline(0, color='#C0392B', linestyle='--', linewidth=2, label='Zero Error (Residual = 0)')
ax.axvline(np.mean(residuals), color='#2980B9', linestyle=':', linewidth=2, label=f'Mean Error = {np.mean(residuals):.2f}')
ax.set_title('Figure 10: Distribution of Prediction Residuals (Errors = Actual - Predicted)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Residual (G3 Actual - G3 Predicted)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig10_residual_distribution.png'), dpi=300)
plt.close()
print("Saved fig10_residual_distribution.png")

# FIGURE 11: Residuals vs Predicted Values (Homoscedasticity Analysis)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(y_pred_mlp_B, residuals, color='#8E44AD', edgecolors='#2C3E50', s=60, alpha=0.8)
ax.axhline(0, color='#C0392B', linestyle='--', linewidth=2, label='Zero Error Line')
ax.set_title('Figure 11: Residuals vs Predicted Final Grades (Homoscedasticity & Bias Check)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Final Grade G3', fontsize=12)
ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
ax.legend(frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig11_residuals_vs_predicted.png'), dpi=300)
plt.close()
print("Saved fig11_residuals_vs_predicted.png")

# FIGURE 12: Permutation Feature Importance for MLP
print("Computing Permutation Feature Importance...")
base_score = mean_squared_error(y_test, y_pred_mlp_B)
perm_importances = []

for col_idx in range(X_test_B.shape[1]):
    X_perm = X_test_B.copy()
    np.random.shuffle(X_perm[:, col_idx])
    y_perm_pred_sc = mlp_B.predict(X_perm, verbose=0).flatten()
    y_perm_pred = y_perm_pred_sc * y_std + y_mean
    perm_mse = mean_squared_error(y_test, y_perm_pred)
    perm_importances.append(perm_mse - base_score)

perm_df = pd.DataFrame({
    'Feature': all_feature_names_B,
    'Importance': perm_importances
}).sort_values(by='Importance', ascending=False)

top_10 = perm_df.head(10).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_10['Feature'], top_10['Importance'], color='#2980B9', edgecolor='#1B4F72')
ax.set_title('Figure 12: Top 10 Most Influential Features (Permutation Importance on Test Set)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Increase in Mean Squared Error (MSE) When Feature is Shuffled', fontsize=12)
ax.set_ylabel('Feature Name', fontsize=12)

for bar in bars:
    width = bar.get_width()
    if width > 0:
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left', va='center', fontsize=9, weight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig12_permutation_importance.png'), dpi=300)
plt.close()
print("Saved fig12_permutation_importance.png")

# Save all results into json
results = {
    "dataset_info": {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx)
    },
    "g3_stats": g3_stats,
    "numerical_summary": num_summary,
    "model_results": {
        "dummy_baseline": {
            "mae": round(mae_dummy, 3),
            "mse": round(mse_dummy, 3),
            "rmse": round(rmse_dummy, 3),
            "r2": round(r2_dummy, 3)
        },
        "linear_reg_setting_A_without_g1_g2": {
            "mae": round(mae_lin_A, 3),
            "mse": round(mse_lin_A, 3),
            "rmse": round(rmse_lin_A, 3),
            "r2": round(r2_lin_A, 3)
        },
        "linear_reg_setting_B_with_g1_g2": {
            "mae": round(mae_lin_B, 3),
            "mse": round(mse_lin_B, 3),
            "rmse": round(rmse_lin_B, 3),
            "r2": round(r2_lin_B, 3)
        },
        "mlp_setting_A_without_g1_g2": {
            "mae": round(mae_mlp_A, 3),
            "mse": round(mse_mlp_A, 3),
            "rmse": round(rmse_mlp_A, 3),
            "r2": round(r2_mlp_A, 3)
        },
        "mlp_setting_B_with_g1_g2": {
            "mae": round(mae_mlp_B, 3),
            "mse": round(mse_mlp_B, 3),
            "rmse": round(rmse_mlp_B, 3),
            "r2": round(r2_mlp_B, 3)
        }
    },
    "top_5_permutation_features": perm_df.head(5).to_dict(orient='records')
}

with open(os.path.join(BASE_DIR, "experiment_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\nAll experiments completed successfully! Results updated in experiment_results.json.")
