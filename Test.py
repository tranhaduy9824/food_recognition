import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the trained model
model = load_model('final_modeldk.h5')

# Define test data path
test_dir = r'F:\Workspace\food_recognition\Training'

# Initialize ImageDataGenerator for the test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create test generator
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Multiclass classification
    shuffle=False  # Don't shuffle so that we can match predictions to filenames
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.2f}')

# Get the true labels and predicted labels
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Predict the labels for the test set and get probabilities
predictions = model.predict(test_generator)

# Get predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Prepare data for CSV
results = []
for i, filepath in enumerate(test_generator.filepaths):
    true_label = class_labels[true_labels[i]]
    predicted_label = class_labels[predicted_classes[i]]
    prediction_percentage = predictions[i][predicted_classes[i]] * 100

    # Append results to the list
    results.append({
        'Image': os.path.basename(filepath),
        'True Label': true_label,
        'Predicted Label': predicted_label,
        'Prediction Percentage': f'{prediction_percentage:.2f}%'
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('test_results_test.csv', index=False)
print("Results exported to 'test_results.csv'.")

# Calculate the MAE (Mean Absolute Error) and MSE (Mean Squared Error)
mae = mean_absolute_error(true_labels, predicted_classes)
mse = mean_squared_error(true_labels, predicted_classes)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')

# Function to plot test accuracy and loss
def plot_test_results(test_acc, test_loss):
    plt.figure(figsize=(6, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.bar(['Test Accuracy'], [test_acc])
    plt.title('Test Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.bar(['Test Loss'], [test_loss])
    plt.title('Test Loss')

    plt.tight_layout()
    plt.show()

# Plot the test results
plot_test_results(test_acc, test_loss)
