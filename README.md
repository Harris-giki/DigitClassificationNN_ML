<body>
    <div class="container">
        <h1>MNIST Handwritten Digit Classification using Neural Networks</h1>
        <p>This project demonstrates a simple neural network model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The model is trained using a basic feedforward neural network, and the project also explores image preprocessing, model training, prediction, and evaluation techniques.</p>
        <h2>Requirements</h2>
        <p>To run this project, the following dependencies are required:</p>
        <ul>
            <li>Python 3.x</li>
            <li>TensorFlow</li>
            <li>NumPy</li>
            <li>OpenCV</li>
            <li>Matplotlib</li>
            <li>Seaborn</li>
        </ul>
        <p>You can install the required dependencies using pip:</p>
        <pre><code>pip install tensorflow numpy opencv-python matplotlib seaborn</code></pre>
        <h2>Dataset</h2>
        <p>The MNIST dataset consists of 70,000 28x28 grayscale images of handwritten digits (0-9). The dataset is divided into a training set of 60,000 images and a test set of 10,000 images. This project uses these images to train a neural network for the task of handwritten digit classification.</p>
        <h2>Steps</h2>
        <h3>1. Data Preprocessing</h3>
        <p>The dataset is loaded using <code>keras.datasets.mnist</code>. The pixel values of the images are normalized to a range between 0 and 1 by dividing each pixel value by 255.</p>
        <h3>2. Model Building</h3>
        <p>A basic neural network is used for classification:</p>
        <ul>
            <li><strong>Flatten Layer:</strong> The 28x28 image is flattened into a 1D array of 784 features.</li>
            <li><strong>Dense Layers:</strong> Two hidden layers of 50 neurons each with ReLU activation are used.</li>
            <li><strong>Output Layer:</strong> A final output layer with 10 neurons (one for each digit) and sigmoid activation is used to make the classification.</li>
        </ul>
        <h3>3. Model Training</h3>
        <p>The model is trained for 15 epochs using the Adam optimizer and sparse categorical cross-entropy loss function.</p>
        <h3>4. Model Evaluation</h3>
        <p>After training, the model is evaluated on the test set to calculate accuracy and loss.</p>
        <h3>5. Prediction</h3>
        <p>The trained model is used to make predictions on the test images, and the predicted labels are compared to the true labels. The model's predictions are visualized using a confusion matrix.</p>
        <h3>6. Custom Image Prediction</h3>
        <p>A custom image of a handwritten digit is loaded, resized, and converted to grayscale. The image is then scaled and reshaped to match the input format of the model. The model predicts the digit in the image, and the predicted label is displayed.</p>
        <h2>Code Walkthrough</h2>
        <h3>1. Loading Libraries and Data</h3>
        <p>Libraries such as TensorFlow, NumPy, and OpenCV are imported. The MNIST dataset is loaded from Keras.</p>
        <h3>2. Preprocessing</h3>
        <p>Data is scaled to the range [0, 1] by dividing by 255. The first image in the training set is displayed using Matplotlib.</p>
        <h3>3. Model Architecture</h3>
        <p>The model consists of three layers: Flatten, Dense (2 hidden layers), and Output layer. <code>relu</code> activation is used for hidden layers, and <code>sigmoid</code> for the output layer.</p>
        <h3>4. Training the Model</h3>
        <p>The model is trained on the training set with a specified number of epochs.</p>
        <h3>5. Model Evaluation</h3>
        <p>The model's accuracy is calculated on the test set.</p>
        <h3>6. Prediction and Confusion Matrix</h3>
        <p>The model's predictions are visualized with a confusion matrix using Seaborn.</p>
        <h3>7. Custom Image Prediction</h3>
        <p>A custom image is loaded, resized, converted to grayscale, and reshaped for prediction by the model.</p>
        <h2>Results</h2>
        <p>The model achieves high accuracy on the MNIST test set. The confusion matrix provides insights into where the model may have misclassified the digits. The model is capable of making predictions on custom handwritten digit images.</p>
        <h2>Future Work</h2>
        <p>Possible future improvements include:</p>
        <ul>
            <li><strong>Model Improvement:</strong> Experiment with more advanced models, such as Convolutional Neural Networks (CNNs), to improve accuracy.</li>
            <li><strong>Real-time Prediction:</strong> Implement real-time digit recognition using a camera feed.</li>
        </ul>
        <h2>License</h2>
        <p>This project is licensed under the MIT License.</p>
        <h2>Acknowledgements</h2>
        <ul>
            <li><a href="http://yann.lecun.com/exdb/mnist/">MNIST Dataset</a></li>
            <li>TensorFlow & Keras for providing a high-level deep learning framework.</li>
            <li>OpenCV for image preprocessing.</li>
        </ul>
    </div>
</body>
