import numpy as np

datasets = np.array(range(1, 41)).reshape(10, 4)
print(datasets)

x_data = datasets[:, :3]
y_data = datasets[:, -1]
print(x_data, y_data)
print(x_data.shape, y_data.shape)

timesteps = 3

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_data = split_x(x_data, timesteps)
print(x_data)
print(x_data.shape)

y_data = y_data[timesteps:]
print(y_data)
print(y_data.shape)