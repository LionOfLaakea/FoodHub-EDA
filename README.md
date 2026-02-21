# FoodHub Exploratory Data Analysis

A comprehensive exploratory data analysis (EDA) of FoodHub's order-level data, uncovering key patterns in customer behavior, cuisine preferences, and delivery performance in New York's competitive food delivery market.

## 📋 Project Overview

FoodHub is a food aggregator company (similar to DoorDash or UberEats) operating in New York. This analysis examines order data to understand:
- Customer ordering patterns and preferences
- Popular cuisines and their pricing dynamics
- Delivery performance metrics and optimization opportunities
- Relationships between order characteristics and customer satisfaction

## 📊 Dataset Description

The analysis is based on order-level transaction data containing the following features:

| Feature | Description | Data Type |
|---------|-------------|-----------|
| `order_id` | Unique identifier for each order | Integer |
| `customer_id` | Unique identifier for each customer | Integer |
| `restaurant_name` | Name of the restaurant | String |
| `cuisine_type` | Type of cuisine (American, Japanese, Italian, etc.) | String |
| `cost_of_the_order` | Total cost of the order in dollars | Float |
| `day_of_the_week` | Whether order was placed on a Weekday or Weekend | String |
| `rating` | Customer rating (0-5 stars) | Float |
| `food_preparation_time` | Time taken by restaurant to prepare food (minutes) | Integer |
| `delivery_time` | Time taken to deliver the order (minutes) | Integer |

## 🔍 Key Findings

- **Most Popular Cuisines**: American, Japanese, Italian, and Chinese cuisines dominate order volume, collectively accounting for the majority of FoodHub's business.

- **Weekend Premium Effect**: Weekend orders average 14.9% higher in cost compared to weekday orders ($17.12 vs $14.89), suggesting price elasticity or larger order sizes on weekends.

- **Competitive Delivery Times**: Average delivery time is approximately 24 minutes, demonstrating efficient logistics operations comparable to industry standards.

- **Rating Gap**: Approximately 25% of orders remain unrated, indicating a significant opportunity to improve customer feedback and satisfaction tracking.

- **Preparation Time Impact**: A moderate positive correlation exists between food preparation time and delivery time, suggesting restaurants that take longer to prepare may also be located further from customers or have complex orders.

## 🛠️ Methodology

The analysis follows a structured exploratory data analysis (EDA) framework:

1. **Data Cleaning & Validation**: Loading data, checking for missing values, and understanding data distributions
2. **Univariate Analysis**: Individual examination of each variable including:
   - Distribution plots (histograms, box plots)
   - Descriptive statistics (mean, median, std deviation)
   - Frequency analysis for categorical variables
3. **Bivariate Analysis**: Relationships between variables including:
   - Cost analysis by cuisine type and day of week
   - Delivery time patterns across restaurant types
   - Correlation analysis between numerical variables
4. **Statistical Insights**: Quantitative findings and business implications
5. **Recommendations**: Actionable insights for business strategy

## 💻 Technologies Used

- **Python 3.x** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn
```

### Execution
1. **Prepare the dataset**: Place your FoodHub CSV file in a `data/` directory
2. **Update the data path**: In `FoodHub_EDA.py`, modify the `data_path` variable to point to your dataset:
   ```python
   data_path = 'data/foodhub_orders.csv'  # Update this path
   ```
3. **Run the analysis**:
   ```bash
   python FoodHub_EDA.py
   ```
4. **View outputs**: All visualizations will be saved to the `outputs/` directory

### Expected Output
The script will:
- Print comprehensive statistics and summary tables to console
- Generate 10+ high-resolution visualization PNG files
- Create an `outputs/` directory with all charts and graphs

## 📈 Visualizations Generated

The analysis produces the following key visualizations:

- **Univariate Distributions**: Histograms and box plots for cost, delivery time, prep time, and ratings
- **Cuisine Analysis**: Bar chart showing order volume and average cost by cuisine type
- **Temporal Patterns**: Weekday vs weekend order cost comparison
- **Correlation Analysis**: Scatter plot showing relationship between prep time and delivery time
- **Quality Metrics**: Rating distribution and rated vs unrated order proportions

## 💡 Business Recommendations

### 1. Cuisine Strategy
Focus marketing efforts on the top 4 cuisines (American, Japanese, Italian, Chinese) where customer demand is highest.

### 2. Weekend Pricing Optimization
Implement dynamic pricing or targeted promotions on weekdays to balance demand and increase order frequency.

### 3. Customer Engagement Initiative
Develop strategies to encourage ratings on 25% of unrated orders to improve feedback for service quality improvements.

### 4. Delivery Performance
Maintain current delivery time averages while working to reduce variance and set cuisine-specific expectations.

### 5. Restaurant Partnership Management
Use delivery time and preparation time data to identify optimization opportunities with partner restaurants.

## 📁 Project Structure

```
FoodHub-EDA/
├── FoodHub_EDA.py       # Main analysis script
├── README.md            # This file
├── data/
│   └── foodhub_orders.csv  # Input dataset (not included)
└── outputs/
    ├── univariate_cost.png
    ├── univariate_cuisine.png
    ├── univariate_delivery_time.png
    ├── univariate_prep_time.png
    ├── univariate_rating.png
    ├── univariate_day_of_week.png
    ├── bivariate_cost_cuisine.png
    ├── bivariate_cost_day.png
    ├── bivariate_delivery_cuisine.png
    ├── bivariate_prep_delivery_correlation.png
    └── bivariate_cost_rating.png
```

## 👤 Author

**Jeremy Gracey**
Data Science Professional

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Questions or Feedback?

For questions about this analysis or potential improvements, feel free to reach out or open an issue on GitHub.

**Last Updated**: February 2024
