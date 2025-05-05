#!/usr/bin/env python
# coding: utf-8

# # Traffic Volume Analysis  
# **Date**: May 2025  
# **Author**: ROHINI
# 
# ---
# 
# #### Objective:  
# Analyze traffic patterns using the **Metro Interstate Traffic Volume** dataset and build a predictive model for traffic volume based on time and weather.
# 
# #### Dataset:  
# - **Source**: [Kaggle - Metro Interstate Traffic Volume](https://www.kaggle.com/datasets/uci/metro-interstate-traffic-volume)
# - **Features**: Date, Time, Weather, Traffic Volume

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import numpy as np


# In[3]:


df = pd.read_csv('C:/Users/Dell/Documents/Metro_interstate_Traffic_Volume.csv')



# In[5]:


print(df.head())
print(df.info())


# In[7]:


print(df.columns)


# In[9]:


print(type(df['date_time'][0]))


# In[11]:


df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True, errors='coerce')
print(df['date_time'].dtype)  


# In[13]:


df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.day_name()


# In[15]:


import matplotlib.pyplot as plt

hourly_avg = df.groupby('hour')['traffic_volume'].mean()

plt.figure(figsize=(10, 5))
hourly_avg.plot(kind='bar', color='skyblue')
plt.title('Average Traffic Volume by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Traffic Volume')
plt.grid(True)
plt.xticks(rotation=0)
plt.show()


# In[17]:


df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday'])


# In[19]:


weekend_avg = df.groupby('is_weekend')['traffic_volume'].mean()
print(weekend_avg)


# In[25]:


plt.figure(figsize=(6, 4))
weekend_avg.plot(kind='bar', color=['green', 'purple'])
plt.xticks([0, 1], ['Weekday', 'Weekend'], rotation=0)
plt.title('Average Traffic Volume: Weekday vs Weekend')
plt.ylabel('Traffic Volume')
plt.grid(True)
plt.show()


# In[23]:


weather_avg = df.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False)
print(weather_avg)


# In[27]:


plt.figure(figsize=(10, 5))
weather_avg.plot(kind='bar', color='cornflowerblue')
plt.title('Average Traffic Volume by Weather')
plt.xlabel('Weather Condition')
plt.ylabel('Traffic Volume')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# In[29]:


import matplotlib.pyplot as plt

df['date'] = df['date_time'].dt.date
daily_avg = df.groupby('date')['traffic_volume'].mean()

plt.figure(figsize=(12, 5))
daily_avg.plot()
plt.title('Daily Average Traffic Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Traffic Volume')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[31]:


df = df.set_index('date_time')
monthly_avg = df['traffic_volume'].resample('ME').mean()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
monthly_avg.plot(marker='o', linestyle='-')
plt.title('Monthly Average Traffic Volume')
plt.xlabel('Month')
plt.ylabel('Average Traffic Volume')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[45]:


df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['is_weekend'] = df.index.weekday >= 5
df['weather_code'] = df['weather_main'].astype('category').cat.codes


# In[35]:


features = ['hour', 'day', 'month', 'is_weekend', 'weather_code']
X = df[features]
y = df['traffic_volume']


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[39]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[41]:


from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")


# In[47]:


# Predict traffic on a clear Monday at 8 AM
sample = pd.DataFrame([{
    'hour': 8,
    'day': 10,
    'month': 1,
    'is_weekend': False,
    'weather_code': df[df['weather_main'] == 'Clear']['weather_code'].iloc[0]
}])

predicted_volume = model.predict(sample)
print(f"Predicted traffic volume: {int(predicted_volume[0])}")


# ## Conclusion
# 
# - **Peak Traffic Hours:** Analysis of hourly averages shows the highest traffic volume occurs between 7 AM-9 AM and 4 PM-6 PM.  
# - **Weekday vs. Weekend:** Average traffic is approximately 20% lower on weekends compared to weekdays.  
# - **Weather Impact:** While clear and cloudy conditions dominate, weather has a minimal effect on overall traffic volume.  
# - **Long-Term Trends:** Monthly averages fluctuate modestly, indicating potential seasonal or holiday influences.  
# - **Model Performance:** A Random Forest regressor achieved a Mean Absolute Error (MAE) of **___** on the test set, demonstrating reasonable predictive accuracy for operational use.
# 
# *End of report.*
# 
# 

# In[51]:


import plotly.express as px

fig = px.box(
    df,
    x='weather_main',
    y='traffic_volume',
    points='all',
    title='Traffic Volume Distribution by Weather'
)
fig.update_layout(xaxis_title='Weather', yaxis_title='Traffic Volume')
fig.show()


# In[53]:


fig = px.histogram(
    df,
    x='hour',
    color='weather_main',
    title='Traffic Volume by Hour and Weather',
    barmode='group'
)
fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Traffic Volume')
fig.show()


# ### Dashboard Summary
# 
# - 🚗 Peak traffic hours: 7 AM – 9 AM and 4 PM – 6 PM
# - 🌦️ Weather affects volume: clear weather has higher traffic
# - 📅 Weekends usually show lower volume
# 

# In[ ]:




