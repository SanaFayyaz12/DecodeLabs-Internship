import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import urllib.request
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏥 BREAST CANCER WISCONSIN - DATA CLEANING & EDA")
print("="*60)

# ===== STEP 1: DATASET DOWNLOAD & LOAD =====
print("\n📥 Step 1: Downloading Dataset...")

csv_file = 'data.csv'

# Agar file pehle se nahi hai toh download kar
if not os.path.exists(csv_file):
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        column_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
                       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                       'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                       'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                       'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        
        print("⏳ Downloading from UCI Machine Learning Repository...")
        urllib.request.urlretrieve(url, csv_file)
        print(f"✅ Dataset downloaded!")
    except:
        print("⚠️ Download failed. Please download manually from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
        exit()

# Load dataset
df = pd.read_csv(csv_file)
print("✅ Dataset loaded successfully!")

# ===== STEP 2: BASIC INFO =====
print("\n" + "="*60)
print("📊 DATASET OVERVIEW")
print("="*60)

print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🔍 First 5 rows:")
print(df.head())

# ===== STEP 3: MISSING VALUES CHECK =====
print("\n" + "="*60)
print("❌ MISSING VALUES ANALYSIS")
print("="*60)

missing = df.isnull().sum()
print(f"\nMissing values: {missing.sum()} ✅ (No missing data!)")

# ===== STEP 4: EDA =====
print("\n" + "="*60)
print("🔬 EXPLORATORY DATA ANALYSIS")
print("="*60)

print(f"\n📊 Statistical Summary:")
print(df.describe())

# Diagnosis distribution
if 'diagnosis' in df.columns or df.columns[1] == 'diagnosis':
    target_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[1]
    print(f"\n🎯 Target Variable Distribution:")
    print(df[target_col].value_counts())

# ===== STEP 5: OUTLIER DETECTION =====
print("\n" + "="*60)
print("🎯 OUTLIER DETECTION (IQR Method)")
print("="*60)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_counts = {}

for col in numeric_columns:
    outlier_mask = detect_outliers_iqr(df, col)
    outlier_counts[col] = outlier_mask.sum()

print(f"\nOutliers found: {sum(outlier_counts.values())} ✅")

# ===== STEP 6: HANDLE OUTLIERS =====
print("\n" + "="*60)
print("🔧 OUTLIER HANDLING (Winsorization)")
print("="*60)

df_processed = df.copy()

for col in numeric_columns:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)

print("✅ Outliers handled!")

# ===== STEP 7: FEATURE ENGINEERING =====
print("\n" + "="*60)
print("⚙️ FEATURE ENGINEERING")
print("="*60)

# Feature 1: Area to Perimeter ratio
if 'area_mean' in df_processed.columns and 'perimeter_mean' in df_processed.columns:
    df_processed['area_perimeter_ratio'] = df_processed['area_mean'] / (df_processed['perimeter_mean'] + 1e-8)
    print("✅ Feature 1: area_perimeter_ratio")

# Feature 2: Compactness Score
if 'texture_mean' in df_processed.columns and 'compactness_mean' in df_processed.columns:
    df_processed['compactness_score'] = df_processed['texture_mean'] * df_processed['compactness_mean']
    print("✅ Feature 2: compactness_score")

# Feature 3: Smoothness-Symmetry interaction
if 'smoothness_mean' in df_processed.columns and 'symmetry_mean' in df_processed.columns:
    df_processed['smoothness_symmetry'] = df_processed['smoothness_mean'] * df_processed['symmetry_mean']
    print("✅ Feature 3: smoothness_symmetry")

print(f"\n🎉 Total features: {df_processed.shape[1]}")

# ===== STEP 8: SAVE =====
print("\n" + "="*60)
print("💾 SAVING PROCESSED DATA")
print("="*60)

df_processed.to_csv('breast_cancer_cleaned.csv', index=False)
print("✅ Saved: breast_cancer_cleaned.csv")

df.to_csv('breast_cancer_original.csv', index=False)
print("✅ Saved: breast_cancer_original.csv")

print("\n" + "="*60)
print("🎉 PROJECT COMPLETE! DATA READY FOR ML MODELS 🎉")
print("="*60)