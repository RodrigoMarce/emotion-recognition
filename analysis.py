import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# takes input file from user
file = input("Input file: ")

# fixes directory path
to_analyze = os.path.join('PEACE', file, 'table2.csv')

# reads csv file into numpy array
table = pd.read_csv(to_analyze)
table = table.to_numpy()

n = len(table)

# takes averages of arousal and valence for the first and last thirds of the data
avg_ar_first = np.mean(table[:n//3, 2])
avg_ar_last = np.mean(table[n//3:, 2])

avg_val_first = np.mean(table[:n//3, 3])
avg_val_last = np.mean(table[(2*(n//3)):, 3])

# creates new table to store the results
result = np.array([[avg_ar_first, avg_val_first], [avg_ar_last, avg_val_last]])

# sets labels for the resulting table
x_labels = ['arousal', 'valence']
y_labels = ['first', 'last']

# creates new table with data and labels
result = pd.DataFrame(result, index=y_labels, columns=x_labels)

# sets display format for 3 decimal numbers
result = result.round(5)

# makes sure the new table gets stored in the same directory as the other data
input_dir = os.path.dirname(to_analyze)
output_file = os.path.join(input_dir, 'analysis.csv')

# stores the new table as a csv file
result.to_csv(output_file)

# PLOTTING
frame = table[:, 0]
arousal = table[:, 2]
valence = table[:, 3]

# create plot for arousal
plt.plot(frame, arousal)

# create title and labels
plt.title('Frame vs Arousal')
plt.xlabel('Frame')
plt.ylabel('Arousal')

# correct path so graph saves in the same folder
ar_graph = os.path.join('PEACE', file, 'arousal.png')

# save file
plt.savefig(ar_graph)
plt.close()

# create plot for valence
plt.plot(frame, valence)

# create titel and labels
plt.title('Frame vs Valence')
plt.xlabel('Frame')
plt.ylabel('Valence')

# correct path so graph saves in the same folder
val_graph = os.path.join('PEACE', file, 'valence.png')

# save file
plt.savefig(val_graph)
plt.close()

# Moving average 
window_size = 5000
ar_moving_avg = np.convolve(arousal, np.ones(window_size)/window_size, mode='valid')
val_moving_avg = np.convolve(valence, np.ones(window_size)/window_size, mode='valid')

plt.plot(frame[:len(ar_moving_avg)], ar_moving_avg, label='Arousal', color='red')
plt.plot(frame[:len(val_moving_avg)], val_moving_avg, label='Valence', color='blue')
plt.legend()

# Linear Regression
frame_reshaped = frame.reshape(-1, 1)

model1 = LinearRegression()
model1.fit(frame_reshaped, arousal)
ar_trend = model1.predict(frame_reshaped)

model2 = LinearRegression()
model2.fit(frame_reshaped, valence)
val_trend = model2.predict(frame_reshaped)

plt.plot(frame, ar_trend, color='red')
plt.plot(frame, val_trend, color='blue')

plt.savefig(os.path.join('PEACE', file, 'smooth.png'))
