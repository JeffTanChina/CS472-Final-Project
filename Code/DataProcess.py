import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Read CSV files
data = pd.read_csv('weatherAUS.csv')

# Turn all columns with key "data" into integer 0
data['Date'] = 0
data['Location'] = 0

# Map a specific string to an integer
data = data.replace({'NNE': 1, 'NE': 2, 'ENE': 3, 'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7, 'S': 8, 'SSW': 9,
                     'SW': 10, 'WSW': 11, 'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15, 'N': 16})

# Map the value in the "RainToday" column to an integer
data['RainToday'] = data['RainToday'].replace({'No': 0, 'Yes': 1})

# Map the value in the "RainTomorrow" column to an integer
data['RainTomorrow'] = data['RainTomorrow'].replace({'No': -1, 'Yes': 1})

# Delete duplicate rows and keep one of them
data = data.drop_duplicates()

# Data splitting by "Location" column
train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['Location'])

train_data = train_data.fillna(train_data.mean())
test_data = test_data.fillna(test_data.mean())

train_data = train_data[(train_data['RainTomorrow'] == 1) | (train_data['RainTomorrow'] == -1)]
test_data = test_data[(test_data['RainTomorrow'] == 1) | (test_data['RainTomorrow'] == -1)]

# Split data to two CSV files
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = data.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = data.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = data.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = data.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
plt.show()

# Display the boxplots
plt.show()
