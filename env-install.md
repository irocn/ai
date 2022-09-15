1. Anaconda install
a. download form https://www.anaconda.com/products/distribution
b. install python3.8+
c. pip install jupyter pandas matplotlib seaborn tensorflow

Jupyter Notebook, allows you to write live code and share the notebooks with others. You will be using it a lot going forward.
Pandas, is a python data analysis library. The key data structure is a DataFrame.
Matplotlib, and Seaborn are data visualization libraries and are very effective for exploratory data analysis.
Tensorflow, is a machine learning and deep learning library developed by Google. By default, it will install version 2 of tensorflow. 
Note that there are slight differences b/w version 1 and version 2 of tensorflow. I suggest, you work with latest version.
Keras, provides high level APIs and comes integrated with tensorflow 2. Its a API wrapper built on top of tensorflow.

2. setting up environment
conda create -n helloworld python==3.9.0
conda activate helloworld

3. Dataset
a. MNIST Dataset

4. Code

# Step 1. Load train and test data set.
mnist = tf.keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

##Scale the data and create validation set: The next step is to scale the data b/w 0 and 1 and create the validation dataset. 
##For the validation dataset, we will divide the X_train_full, y_train_full into two sets of X_valid, X_train and y_valid, y_train.
# Scale the data b/w 0 and 1 by dividing it by 255 as its unsigned int
X_train_full = X_train_full/255.
X_test = X_test/255.

# Create the validation data from training data.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_train.shape
# should give o/p of (55000, 28, 28)
X_valid.shape
# should give o/p of (5000, 28, 28)


# view the actual image at index 30
plt.imshow(X_train[30], cmap='binary')

# Lets look at the pixels in detail using SNS
plt.figure(figsize=(15,15))
sns.heatmap(X_train[30], annot=True, cmap='binary')

Model Building: Its now time to build our model. The concepts from my first article will be useful to understand it better. 
Here is the code to build the model.

# lets create the model
# Flatten = make the array to sequential layer
# Dense = creating a hidden OR output layer
LAYERS = [tf.keras.layers.Flatten(input_shape=[28,28],
name="inputLayer"),
         tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
         tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
         tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")]
model = tf.keras.models.Sequential(LAYERS)

