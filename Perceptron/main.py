import numpy as np
import Perceptron as prc
import matplotlib.pyplot as plt

#Dataset
data = np.genfromtxt('data/dataset_2.txt', delimiter=' ')
X_train, y_train = data[:, :2], data[:, 2]
y_train = y_train.astype(np.int)

#Graphic plotting
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')
plt.title('Training set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.show()

#Training the Perceptron
ppn = prc.Perceptron()
ppn.train(X_train, y_train, epochs=50)
print("Parameters of model:\t")
print('Weights:%s\n' % ppn.weights)
print('Bias: %s\n' % ppn.bias)

#Evaluating the model
train_acc = ppn.evaluate(X_train, y_train)
print('Accuracy: %s' % (train_acc*100))


#Testing set
data = np.genfromtxt('data/train_set.txt', delimiter=' ')


#Plotling training and testing sets
fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))
x_min, x_max = 0, 1
ax[1].plot([x_min, x_max], [x_max, x_max])
ax[1].set_title('Testing set')
ax[1].get_legend()
ax[1].grid()
for x, y in data:
    if (np.dot((ppn.weights).T, np.array([x, y]))+ppn.bias) >0:
        ax[1].scatter(x, y, label='class 1', marker='s', color='orange')
    else:
        ax[1].scatter(x, y, label='class 0', marker='o', color='blue')


ax[0].plot([x_min, x_max], [x_max, x_max])
ax[0].set_title('Training set')
ax[0].get_legend()
ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='class 0', marker='o', color='blue')
ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='class 1', marker='s', color='orange')
ax[0].grid()

plt.show()




















