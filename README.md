# Machine-Learning-Problems
A repository for machine learning problems and exploration of different ML libraries. The goal of this repository is to collect takeaways while developing ML models. This should improve my overall understanding of developing machine learning applications.

-----

## Takeways

#### 01_MNIST
1. One of the first takeaways is that it is important what kind of loss function you are using in this specific problem. You can either choose the sparse_categorical_crossentropy loss or the categorical_crossentropy loss. You may have noticed that we used these two different loss functions for both datasets. The main difference is how we specify and shape our tensors that contain the labels for the specific images. For example, if you use the sparse categorical crossentropy loss the network expects to get a single value at the final output layer to calculate the error. This value is an integer which denotes the different output classes. If you want to use the standard categorical crossentropy loss you need to one-hot-encode the labels. you can do that with the following code snippet.

```python
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
```

2. If you want to test a new neural network architecture or a new deep learning framework it makes sense to use the Keras datasets. This will allow you to concentrate on the architecture and do not care about the data preprocessing step. Since preprocessing of images and might take a lot of time this will allow you to focus on the architecture. You can use the Keras datasets by invoking the following peace of code. Keras provides you with the following datasets: CIFAR10, CIFAR100, IMDB movie review sentiments, Reuters news wirestopics, MNIST, Fashion MNIST and Boston housing prices.

```python
dataset = keras.datasets.<YOUR_DATASET>
(train_input, train_labels), (test_input, test_labels) = dataset.load_data()
```

3. It makes sense to take a look at the output of the artificial neural network. Especially if your neural network does not perform very well this might help you to indicate which part of the network you might want to improve. In our case we can see that the digits in the Number MNIST dataset are simply unreadable or even look more like another number than the actual digit. For the Fashion MNIST dataset we can see that some items (e.g. Ankle Boot) look like another one (e.g. Sandal) because they share some common characteristic. It would be interesting to see if a bigger network can handle different types of fashion items in a more efficient way?!

#### 02_RNNs
1. You should always capture the training history including your training and validation performance. This will allow you to plot the improvement of accuracy and loss over time. It is important to do that because you will be able to recognize if your model overfits and maybe the learning rate is too high. Additionally, you need to pass the validation data to the model.fit function in order to obtain validation results. However, you can also specify the validation split so that Keras knows how much of the training data it should set aside in order to use that data for validation. The following code snippets show how you can set the validation data and record the history properly:
```python
history = model.fit(x_train,y_train, epochs = 4, batch_size= 32, validation_data=(x_test, y_test))
history = model.fit(x_train,y_train, epochs = 4, batch_size= 32, validation_split=0.2)
```

2. If you try to build a multilayer RNNs you probably need to set the `return_sequences` parameter for LSTM or CuDNNLSTM Layers to `True`. This is because the following layer expects to get an input from each timestep. For example, if you look at a time series of 80 time steps this parameter will make sure that there are 80 hidden values passed to the next layer. It's quite hard to explain so just test it out :)

#### 03_PyTorch
1. You need to specify the batch size in the dataloader. Compared to Keras and Tensorflow which allow to specify the batch_size in the fit/evaluate/predict function this is a big difference. However, if there is no dataloader you can probably also specify the batch size in your training/testing function. Instead of using the following line of code you can probably define your own way of feeding the data into the network.
```python
for batch_idx, (data, target) in enumerate(train_loader):
```

2. You can add prints everywhere which makes it so much easier to debug the whole network and its dimensions. ALthough some of the error messages are hard to understand you can always add print statements in your forward pass of your network module. In fact, you only need to add the following line to understand the dimensions of your network.
```python
print(x.shape)
```
3. You can customize the training loop very strongly. As already said, it seems to be very tricky and error prone to define your own training and testing loop. Nonetheless, I think that you can probably copy paste these functions to every new project to make sure to not introduce unnecessary bugs. Additionally, this process allows you to customize a lot of things. For instance, if you have a special way of keeping track of your losses you can do this with PyTorch.
