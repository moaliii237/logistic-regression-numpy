import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importind data from my desktop
data = pd.read_csv('./data/diabetic_data.csv')

#view columns names, number and basic info
print(data.head())
print(data.columns)
print(len(data.columns))
print(data.info())

# bar plot for a categorical variable like (age)

sns.histplot(data['age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# Bar plot for 'gender'

print(data['gender'].value_counts())
print(data['age'].value_counts())

sns.countplot(x='gender', data=data)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Check for missing values
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100

# Display columns with missing values
print(missing_values[missing_values > 0])  # Only columns with missing values
print(missing_percentage[missing_percentage > 0])  # Percentages

