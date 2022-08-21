# first neural network with keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

# load the dataset
dataset = loadtxt('smaller.csv', delimiter=',', encoding='utf-8-sig')
# split into input (X) and output (y) variables
independent_vars = 12;

X = dataset[:, 0:independent_vars]
y = dataset[:, independent_vars]

# define the keras model - multilayer perceptron
num_input = 12
num_hidden = 30
num_output = 1


model = Sequential()
model.add(Dense(num_input, activation='relu'))  # input layer
model.add(Dense(num_hidden, activation='relu'))  # hidden layer
model.add(Dense(num_output, activation='linear'))  # output layer
model.add(LeakyReLU(alpha=0.9))

# Gradient descent algorithm
adam = Adam(0.9)
model.compile(loss='mean_squared_logarithmic_error', optimizer=adam)

# fit the keras model on the dataset
model.fit(X, y, epochs=50, batch_size=5)


# evaluate the keras model
results = model.evaluate(X, y, batch_size=5)

# make probability predictions with the model
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
# summarize the first 5 cases
for i in range(5):
	print(X[i].tolist())
	print("predicted: ", predictions[i])
	print("expected: ", y[i])
