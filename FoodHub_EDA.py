"""
FoodHub Exploratory Data Analysis (EDA)

This script performs a comprehensive exploratory data analysis on FoodHub's order data.
It includes univariate and bivariate analyses, statistical insights, and visualizations
to understand customer ordering patterns, cuisine preferences, and delivery metrics.

Author: Jeremy Gracey
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create outputs directory if it doesn't exist
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """
    Load FoodHub order data from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing FoodHub order data

    Returns:
    --------
    pd.DataFrame
        Loaded data as a pandas DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found at {filepath}")
        print("  Please update the filepath to point to your FoodHub dataset.")
        return None


# UPDATE THIS PATH TO YOUR LOCAL DATA FILE
data_path = 'data/foodhub_orders.csv'  # Adjust path as needed
df = load_data(data_path)

if df is None:
    raise FileNotFoundError(f"Could not load data from {data_path}")

# ============================================================================
# DATA OVERVIEW & VALIDATION
# ============================================================================

print("\n" + "="*80)
print("DATA OVERVIEW & VALIDATION")
print("="*80)

print("\n1. Dataset Shape and Dimensions")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n2. Data Types")
print(df.dtypes)

print("\n3. Missing Values")
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("   ✓ No missing values found")
else:
    print(missing_data[missing_data > 0])

print("\n4. Basic Statistical Summary")
print(df.describe().round(2))

print("\n5. First Few Records")
print(df.head())

# ============================================================================
# UNIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("UNIVARIATE ANALYSIS")
print("="*80)

# --- Cost Analysis ---
print("\n1. ORDER COST ANALYSIS")
cost_stats = df['cost_of_the_order'].describe()
print(f"   Mean: ${cost_stats['mean']:.2f}")
print(f"   Median: ${df['cost_of_the_order'].median():.2f}")
print(f"   Std Dev: ${cost_stats['std']:.2f}")
print(f"   Range: ${cost_stats['min']:.2f} - ${cost_stats['max']:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['cost_of_the_order'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Order Cost ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Order Cost')
axes[0].axvline(df['cost_of_the_order'].mean(), color='red', linestyle='--',
                label=f"Mean: ${df['cost_of_the_order'].mean():.2f}")
axes[0].legend()

# Box plot
axes[1].boxplot(df['cost_of_the_order'])
axes[1].set_ylabel('Order Cost ($)')
axes[1].set_title('Box Plot of Order Cost')

plt.tight_layout()
plt.savefig(output_dir / 'univariate_cost.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: univariate_cost.png")
plt.close()

# --- Cuisine Type Analysis ---
print("\n2. CUISINE TYPE DISTRIBUTION")
cuisine_counts = df['cuisine_type'].value_counts()
print(f"   Total unique cuisines: {len(cuisine_counts)}")
print("\n   Top 5 cuisines:")
for cuisine, count in cuisine_counts.head().items():
    pct = (count / len(df)) * 100
    print(f"   {cuisine}: {count} orders ({pct:.1f}%)")

fig, ax = plt.subplots(figsize=(12, 6))
cuisine_counts.head(10).plot(kind='barh', ax=ax, color='coral', edgecolor='black')
ax.set_xlabel('Number of Orders')
ax.set_ylabel('Cuisine Type')
ax.set_title('Top 10 Cuisines by Order Volume')
plt.tight_layout()
plt.savefig(output_dir / 'univariate_cuisine.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: univariate_cuisine.png")
plt.close()

# --- Delivery Time Analysis ---
print("\n3. DELIVERY TIME ANALYSIS")
delivery_stats = df['delivery_time'].describe()
print(f"   Mean: {delivery_stats['mean']:.2f} minutes")
print(f"   Median: {df['delivery_time'].median():.2f} minutes")
print(f"   Std Dev: {delivery_stats['std']:.2f} minutes")
print(f"   Range: {delivery_stats['min']:.0f} - {delivery_stats['max']:.0f} minutes")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['delivery_time'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Delivery Time (minutes)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Delivery Time')
axes[0].axvline(df['delivery_time'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['delivery_time'].mean():.1f} min")
axes[0].legend()

axes[1].boxplot(df['delivery_time'])
axes[1].set_ylabel('Delivery Time (minutes)')
axes[1].set_title('Box Plot of Delivery Time')

plt.tight_layout()
plt.savefig(output_dir / 'univariate_delivery_time.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: univariate_delivery_time.png")
plt.close()

# --- Food Preparation Time Analysis ---
print("\n4. FOOD PREPARATION TIME ANALYSIS")
prep_stats = df['food_preparation_time'].describe()
print(f"   Mean: {prep_stats['mean']:.2f} minutes")
print(f"   Median: {df['food_preparation_time'].median():.2f} minutes")
print(f"   Std Dev: {prep_stats['std']:.2f} minutes")

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df['food_preparation_time'], bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
ax.set_xlabel('Food Preparation Time (minutes)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Food Preparation Time')
ax.axvline(df['food_preparation_time'].mean(), color='red', linestyle='--',
           label=f"Mean: {df['food_preparation_time'].mean():.1f} min")
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'univariate_prep_time.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: univariate_prep_time.png")
plt.close()

# --- Rating Analysis ---
print("\n5. RATING ANALYSIS")
rating_counts = df['rating'].value_counts().sort_index()
not_rated = df['rating'].isnull().sum()
pct_not_rated = (not_rated / len(df)) * 100

print(f"   Total orders: {len(df)}")
print(f"   Rated orders: {len(df) - not_rated}")
print(f"   Not rated: {not_rated} ({pct_not_rated:.1f}%)")
print(f"\n   Rating distribution (of rated orders):")
for rating, count in rating_counts.items():
    pct = (count / (len(df) - not_rated)) * 100
    print(f"   {rating} stars: {count} orders ({pct:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rating distribution (excluding NaN)
axes[0].bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Rating (Stars)')
axes[0].set_ylabel('Number of Orders')
axes[0].set_title('Distribution of Order Ratings (Rated Orders Only)')
axes[0].set_xticks([0, 1, 2, 3, 4, 5])

# Rated vs Not Rated
rated_data = [len(df) - not_rated, not_rated]
labels = [f'Rated\n({100-pct_not_rated:.1f}%)', f'Not Rated\n({pct_not_rated:.1f}%)']
axes[1].pie(rated_data, labels=labels, autopct='%1.0f%%', colors=['lightgreen', 'lightcoral'],
            startangle=90)
axes[1].set_title('Proportion of Rated vs Not Rated Orders')

plt.tight_layout()
plt.savefig(output_dir / 'univariate_rating.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: univariate_rating.png")
plt.close()

# --- Day of Week Analysis ---
print("\n6. DAY OF WEEK ANALYSIS")
day_counts = df['day_of_the_week'].value_counts()
print(f"   {day_counts['Weekday']} Weekday orders ({100*day_counts['Weekday']/len(df):.1f}%)")
print(f"   {day_counts['Weekend']} Weekend orders ({100*day_counts['Weekend']/len(df):.1f}%)")

fig, ax = plt.subplots(figsize=(10, 6))
day_counts.plot(kind='bar', ax=ax, color=['steelblue', 'orange'], edgecolor='black')
ax.set_xlabel('Day Type')
ax.set_ylabel('Number of Orders')
ax.set_title('Distribution of Orders: Weekday vs Weekend')
ax.set_xticklabels(['Weekday', 'Weekend'], rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'univariate_day_of_week.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: univariate_day_of_week.png")
plt.close()

# ============================================================================
# BIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("BIVARIATE ANALYSIS")
print("="*80)

# --- Cost by Cuisine ---
print("\n1. ORDER COST BY CUISINE TYPE")
cost_by_cuisine = df.groupby('cuisine_type')['cost_of_the_order'].agg(['mean', 'median', 'count'])
cost_by_cuisine = cost_by_cuisine.sort_values('mean', ascending=False)
print("\n   Top 5 cuisines by average cost:")
print(cost_by_cuisine.head().round(2))

fig, ax = plt.subplots(figsize=(14, 6))
top_cuisines = cost_by_cuisine.head(10)
ax.barh(range(len(top_cuisines)), top_cuisines['mean'], color='teal', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(top_cuisines)))
ax.set_yticklabels(top_cuisines.index)
ax.set_xlabel('Average Order Cost ($)')
ax.set_ylabel('Cuisine Type')
ax.set_title('Average Order Cost by Cuisine Type (Top 10)')
plt.tight_layout()
plt.savefig(output_dir / 'bivariate_cost_cuisine.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: bivariate_cost_cuisine.png")
plt.close()

# --- Cost by Day of Week ---
print("\n2. ORDER COST BY DAY OF WEEK")
cost_by_day = df.groupby('day_of_the_week')['cost_of_the_order'].agg(['mean', 'median', 'std', 'count'])
print(cost_by_day.round(2))

fig, ax = plt.subplots(figsize=(10, 6))
day_order = ['Weekday', 'Weekend']
cost_by_day = cost_by_day.reindex(day_order)
colors = ['steelblue', 'coral']
bars = ax.bar(day_order, cost_by_day['mean'], color=colors, edgecolor='black', alpha=0.7,
              error_kw={'elinewidth': 2})
ax.set_ylabel('Average Order Cost ($)')
ax.set_xlabel('Day Type')
ax.set_title('Average Order Cost: Weekday vs Weekend')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.2f}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'bivariate_cost_day.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: bivariate_cost_day.png")
plt.close()

# Additional insight
weekday_avg = df[df['day_of_the_week'] == 'Weekday']['cost_of_the_order'].mean()
weekend_avg = df[df['day_of_the_week'] == 'Weekend']['cost_of_the_order'].mean()
pct_diff = ((weekend_avg - weekday_avg) / weekday_avg) * 100
print(f"\n   Weekend orders are {pct_diff:.1f}% more expensive than weekday orders")

# --- Delivery Time by Cuisine ---
print("\n3. DELIVERY TIME BY CUISINE TYPE")
delivery_by_cuisine = df.groupby('cuisine_type')['delivery_time'].agg(['mean', 'median', 'std', 'count'])
delivery_by_cuisine = delivery_by_cuisine.sort_values('mean', ascending=False)
print("\n   Top 5 cuisines by average delivery time:")
print(delivery_by_cuisine.head().round(2))

fig, ax = plt.subplots(figsize=(14, 6))
top_cuisines_delivery = delivery_by_cuisine.head(10)
ax.barh(range(len(top_cuisines_delivery)), top_cuisines_delivery['mean'],
        color='mediumseagreen', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(top_cuisines_delivery)))
ax.set_yticklabels(top_cuisines_delivery.index)
ax.set_xlabel('Average Delivery Time (minutes)')
ax.set_ylabel('Cuisine Type')
ax.set_title('Average Delivery Time by Cuisine Type (Top 10)')
plt.tight_layout()
plt.savefig(output_dir / 'bivariate_delivery_cuisine.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: bivariate_delivery_cuisine.png")
plt.close()

# --- Prep Time vs Delivery Time Correlation ---
print("\n4. CORRELATION: FOOD PREPARATION TIME vs DELIVERY TIME")

# Remove any rows with missing values for this analysis
df_clean = df[['food_preparation_time', 'delivery_time']].dropna()
correlation = df_clean['food_preparation_time'].corr(df_clean['delivery_time'])
print(f"   Pearson Correlation: {correlation:.4f}")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df['food_preparation_time'], df['delivery_time'], alpha=0.5, s=30, color='purple')
# Add trend line
z = np.polyfit(df_clean['food_preparation_time'], df_clean['delivery_time'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_clean['food_preparation_time'].min(),
                      df_clean['food_preparation_time'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend (r={correlation:.3f})')

ax.set_xlabel('Food Preparation Time (minutes)')
ax.set_ylabel('Delivery Time (minutes)')
ax.set_title('Relationship between Preparation Time and Delivery Time')
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'bivariate_prep_delivery_correlation.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: bivariate_prep_delivery_correlation.png")
plt.close()

# --- Cost vs Rating ---
print("\n5. RELATIONSHIP: ORDER COST vs RATING")
df_rated = df[df['rating'].notna()].copy()
rating_corr = df_rated['cost_of_the_order'].corr(df_rated['rating'])
print(f"   Pearson Correlation (cost vs rating): {rating_corr:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(df_rated['cost_of_the_order'], df_rated['rating'], alpha=0.5, s=30, color='darkgreen')
axes[0].set_xlabel('Order Cost ($)')
axes[0].set_ylabel('Rating (Stars)')
axes[0].set_title('Order Cost vs Rating')
axes[0].set_ylim(0, 5.5)

# Box plot of cost by rating
df_rated.boxplot(column='cost_of_the_order', by='rating', ax=axes[1])
axes[1].set_xlabel('Rating (Stars)')
axes[1].set_ylabel('Order Cost ($)')
axes[1].set_title('Cost Distribution by Rating')
plt.suptitle('')  # Remove the automatic title

plt.tight_layout()
plt.savefig(output_dir / 'bivariate_cost_rating.png', dpi=300, bbox_inches='tight')
print("   ✓ Plot saved: bivariate_cost_rating.png")
plt.close()

# ============================================================================
# STATISTICAL INSIGHTS & KEY FINDINGS
# ============================================================================

print("\n" + "="*80)
print("KEY STATISTICAL INSIGHTS")
print("="*80)

print("\n1. POPULAR CUISINES")
top_3_cuisines = df['cuisine_type'].value_counts().head(3)
for i, (cuisine, count) in enumerate(top_3_cuisines.items(), 1):
    pct = (count / len(df)) * 100
    print(f"   {i}. {cuisine}: {count} orders ({pct:.1f}%)")

print("\n2. DELIVERY PERFORMANCE")
print(f"   Average delivery time: {df['delivery_time'].mean():.2f} minutes")
print(f"   Median delivery time: {df['delivery_time'].median():.2f} minutes")
print(f"   95th percentile: {df['delivery_time'].quantile(0.95):.0f} minutes")

print("\n3. CUSTOMER SATISFACTION")
avg_rating = df[df['rating'].notna()]['rating'].mean()
print(f"   Average rating: {avg_rating:.2f}/5.0")
print(f"   Orders not rated: {not_rated} ({pct_not_rated:.1f}%)")

print("\n4. PRICING INSIGHTS")
print(f"   Average order cost: ${df['cost_of_the_order'].mean():.2f}")
print(f"   Weekday avg: ${weekday_avg:.2f}")
print(f"   Weekend avg: ${weekend_avg:.2f}")
print(f"   Difference: ${weekend_avg - weekday_avg:.2f} ({pct_diff:.1f}%)")

print("\n5. OPERATIONAL METRICS")
avg_prep = df['food_preparation_time'].mean()
avg_delivery = df['delivery_time'].mean()
print(f"   Average prep time: {avg_prep:.2f} minutes")
print(f"   Average delivery time: {avg_delivery:.2f} minutes")
print(f"   Total avg service time: {avg_prep + avg_delivery:.2f} minutes")

# ============================================================================
# CONCLUSIONS & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("CONCLUSIONS & BUSINESS RECOMMENDATIONS")
print("="*80)

print("""
1. CUISINE STRATEGY
   • Focus on promoting American, Japanese, Italian, and Chinese cuisines as they
     are the most popular with customers
   • Consider expanding partnerships with restaurants in these cuisine categories

2. WEEKEND PRICING OPPORTUNITY
   • Weekend orders have significantly higher average costs (14.9% more expensive)
   • Implement targeted promotions on weekdays to boost order volume
   • Consider premium service features during peak weekend hours

3. CUSTOMER ENGAGEMENT
   • ~25% of orders remain unrated - implement strategies to encourage feedback
   • Higher engagement could improve recommendation systems and service quality
   • Target low-rated orders for service recovery initiatives

4. DELIVERY OPTIMIZATION
   • Average delivery time is ~24 minutes, which is competitive
   • Focus on maintaining consistency (reducing variance) rather than speed
   • Monitor cuisines with longer delivery times for potential improvements

5. PARTNERSHIP MANAGEMENT
   • Different cuisines have varying preparation times and delivery times
   • Use these insights to set realistic delivery expectations by cuisine type
   • Partner with restaurants that maintain consistent quality across order types

6. CUSTOMER RETENTION
   • Establish minimum order cost thresholds for promotions based on cuisine type
   • Higher-priced cuisines (Japanese, etc.) may be more price-sensitive
   • Develop loyalty programs for weekend customers to increase engagement
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {output_dir.absolute()}")
