import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Load the dataset
df = pd.read_csv("ecommerce_dataset_updated.csv")

# Displaying rows
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())

df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], format='%d-%m-%Y')

# Remove missing values
df.dropna(inplace=True)

df['Category'] = df['Category'].astype('category').cat.codes

print(df.info())

sns.set_style("whitegrid")

# Histogram (Distribution of discounts)
plt.figure(figsize=(8,5))
sns.histplot(df['Discount (%)'], bins=20, kde=True, color='blue')
plt.title("Distribution of Discounts (%)")
plt.xlabel("Discount Percentage")
plt.ylabel("Frequency")
plt.show()

# Average Discounts Across Product Categories (Bar Chart)
plt.figure(figsize=(10,5))
sns.barplot(x=df.groupby("Category")["Discount (%)"].mean().index, 
            y=df.groupby("Category")["Discount (%)"].mean().values, 
            palette="viridis")
plt.title("Average Discount Across Categories")
plt.xlabel("Category")
plt.ylabel("Average Discount (%)")
plt.xticks(rotation=45)
plt.show()

# Total Sales by Product Category (Bar Chart)
plt.figure(figsize=(10,5))
sns.barplot(x=df.groupby("Category")["Final_Price(Rs.)"].sum().index, 
            y=df.groupby("Category")["Final_Price(Rs.)"].sum().values, 
            palette="magma")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales (Rs.)")
plt.xticks(rotation=45)
plt.show()


# Relationship Between Discounts & Sales (Scatter Plot)
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Discount (%)'], y=df['Final_Price(Rs.)'], alpha=0.6)
plt.title("Discounts vs Final Sales Price")
plt.xlabel("Discount Percentage")
plt.ylabel("Final Price (Rs.)")
plt.show()

#correlation between discounts & final price
print(df[['Discount (%)', 'Final_Price(Rs.)']].corr())


# Linear Regression to predict sales based on discount
X = df[['Discount (%)']]
y = df['Final_Price(Rs.)']

model = LinearRegression()
model.fit(X, y)

print("Regression Coefficient:", model.coef_)





