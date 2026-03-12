# facial_recognition_from_scratch
import numpy as np
from PIL import Image
import os
from random import shuffle

# ---- Step 1: Load images ----


def load_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            img = Image.open(os.path.join(folder_path, file)
                             ).convert('L')  # grayscale
            img = img.resize((50, 50))
            images.append(np.array(img).flatten())
    return images  # return as list


# Load datasets for each person
person1_faces = load_images('/Users/benjukarki/dataset/person1')
person2_faces = load_images('/Users/benjukarki/dataset/person2')

# ---- Step 2: Split into training and testing ----


def train_test_split(images, test_ratio=0.5):
    shuffle(images)
    n_test = max(1, int(len(images)*test_ratio))
    return images[n_test:], images[:n_test]  # training, testing


train1, test1 = train_test_split(person1_faces)
train2, test2 = train_test_split(person2_faces)

# Combine training data
train_dataset = np.vstack([train1, train2])
train_labels = ['person1']*len(train1) + ['person2']*len(train2)

# Combine test data
test_dataset = test1 + test2
test_labels = ['person1']*len(test1) + ['person2']*len(test2)

# ---- Step 3: Recognition function ----


def recognize_face(test_vector, dataset, labels):
    distances = np.linalg.norm(dataset - test_vector, axis=1)
    min_index = np.argmin(distances)
    return labels[min_index]


# ---- Step 4: Test all images ----
correct = 0
for test_vec, true_label in zip(test_dataset, test_labels):
    predicted = recognize_face(test_vec, train_dataset, train_labels)
    print(f"True: {true_label}  Predicted: {predicted}")
    if predicted == true_label:
        correct += 1

accuracy = correct / len(test_dataset)
print(f"\nRecognition accuracy: {accuracy*100:.2f}%")
