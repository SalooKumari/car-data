"""
Car Market Trends Analysis - Car Dekho Dataset from VOIC

Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# STEP 1: Setup
# ---------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------
# STEP 2: Load Data
# ---------------------------------------------------------
df = pd.read_csv("CarDekho_Data.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------
# STEP 3: Data Cleaning
# ---------------------------------------------------------
df = df.drop_duplicates()

# Feature Engineering: Car Age
CURRENT_YEAR = 2020
df["Car_Age"] = CURRENT_YEAR - df["Year"]

# Price depreciation ratio
df["Price_Drop_%"] = ((df["Present_Price"] - df["Selling_Price"]) / df["Present_Price"]) * 100

print("\nAfter cleaning, shape:", df.shape)
print(df[["Car_Name", "Year", "Car_Age", "Selling_Price", "Present_Price", "Price_Drop_%"]].head())

# ---------------------------------------------------------
# STEP 4: Summary Statistics
# ---------------------------------------------------------
print("\nDescriptive statistics:\n", df.describe())
print("\nFuel Type counts:\n", df["Fuel_Type"].value_counts())
print("\nSeller Type counts:\n", df["Seller_Type"].value_counts())
print("\nTransmission counts:\n", df["Transmission"].value_counts())
print("\nOwner counts:\n", df["Owner"].value_counts())

# ---------------------------------------------------------
# STEP 5: Visualizations
# ---------------------------------------------------------

# 5.1 Selling Price Distribution
plt.figure()
sns.histplot(df["Selling_Price"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Selling Price (in Lakhs)")
plt.xlabel("Selling Price (Lakhs)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/01_selling_price_distribution.png", dpi=150)
plt.close()

# 5.2 Fuel Type Distribution
plt.figure()
sns.countplot(data=df, x="Fuel_Type", hue="Fuel_Type", palette="Set2", legend=False)
plt.title("Count of Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig("outputs/02_fuel_type_count.png", dpi=150)
plt.close()

# 5.3 Selling Price vs Present Price
plt.figure()
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Fuel_Type", alpha=0.8)
plt.title("Selling Price vs Present (Ex-Showroom) Price")
plt.xlabel("Present Price (Lakhs)")
plt.ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/03_selling_vs_present_price.png", dpi=150)
plt.close()

# 5.4 Selling Price vs Kilometers Driven
plt.figure()
sns.scatterplot(data=df, x="Kms_Driven", y="Selling_Price", hue="Transmission", alpha=0.8)
plt.title("Selling Price vs Kilometers Driven")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/04_price_vs_kms_driven.png", dpi=150)
plt.close()

# 5.5 Selling Price vs Car Age
plt.figure()
sns.scatterplot(data=df, x="Car_Age", y="Selling_Price", hue="Fuel_Type", alpha=0.8)
plt.title("Car Depreciation: Selling Price vs Car Age")
plt.xlabel("Car Age (Years)")
plt.ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/05_price_vs_car_age.png", dpi=150)
plt.close()

# 5.6 Fuel Type vs Selling Price (Boxplot)
plt.figure()
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", hue="Fuel_Type", palette="Set3", legend=False)
plt.title("Selling Price Distribution by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/06_fuel_type_vs_price_box.png", dpi=150)
plt.close()

# 5.7 Transmission vs Selling Price (Boxplot)
plt.figure()
sns.boxplot(data=df, x="Transmission", y="Selling_Price", hue="Transmission", palette="Pastel1", legend=False)
plt.title("Selling Price Distribution by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/07_transmission_vs_price_box.png", dpi=150)
plt.close()

# 5.8 Seller Type vs Selling Price (Boxplot)
plt.figure()
sns.boxplot(data=df, x="Seller_Type", y="Selling_Price", hue="Seller_Type", palette="Set1", legend=False)
plt.title("Selling Price Distribution by Seller Type")
plt.xlabel("Seller Type")
plt.ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/08_seller_type_vs_price_box.png", dpi=150)
plt.close()

# 5.9 Owner Count vs Average Selling Price
plt.figure()
owner_avg = df.groupby("Owner")["Selling_Price"].mean().reset_index()
sns.barplot(data=owner_avg, x="Owner", y="Selling_Price", hue="Owner", palette="viridis", legend=False)
plt.title("Average Selling Price by Number of Previous Owners")
plt.xlabel("Number of Previous Owners")
plt.ylabel("Average Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/09_owner_vs_avg_price.png", dpi=150)
plt.close()

# 5.10 Correlation Heatmap
plt.figure(figsize=(7, 6))
numeric_cols = df[["Selling_Price", "Present_Price", "Kms_Driven", "Car_Age", "Owner"]]
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("outputs/10_correlation_heatmap.png", dpi=150)
plt.close()

# 5.11 Yearly trend - average selling price by manufacturing year
plt.figure()
year_avg = df.groupby("Year")["Selling_Price"].mean().reset_index()
sns.lineplot(data=year_avg, x="Year", y="Selling_Price", marker="o", color="darkorange")
plt.title("Average Selling Price by Manufacturing Year")
plt.xlabel("Manufacturing Year")
plt.ylabel("Average Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/11_avg_price_by_year.png", dpi=150)
plt.close()

print("\nAll plots saved in the 'outputs' folder.")

# ---------------------------------------------------------
# STEP 6: Key Insights
# ---------------------------------------------------------
print("\n--- KEY INSIGHTS ---")
print("Correlation of Selling Price with other numeric features:")
print(corr["Selling_Price"].sort_values(ascending=False))

most_common_fuel = df["Fuel_Type"].mode()[0]
print(f"\nMost common fuel type: {most_common_fuel}")

avg_price_by_fuel = df.groupby("Fuel_Type")["Selling_Price"].mean().sort_values(ascending=False)
print(f"\nAverage selling price by fuel type:\n{avg_price_by_fuel}")

avg_price_by_transmission = df.groupby("Transmission")["Selling_Price"].mean()
print(f"\nAverage selling price by transmission:\n{avg_price_by_transmission}")

print(f"\nAverage depreciation from present price: {df['Price_Drop_%'].mean():.2f}%")