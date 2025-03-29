from tkinter import *
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim

main = Tk()
main.title("Cyber Threat Detection")
main.geometry("1000x700")

# Add background image
try:
    bg_image = Image.open("background.jpg").resize((1000, 700), Image.LANCZOS)
    bg_image = ImageTk.PhotoImage(bg_image)
    Label(main, image=bg_image).place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Error loading background image: {e}")

le = preprocessing.LabelEncoder()
global filename, X, Y, label_names, scaler

def upload():
    global filename, X, Y, label_names, scaler
    filename = filedialog.askopenfilename()
    if not filename:
        text.insert(END, "No file selected.\n")
        return
    dataset = pd.read_csv(filename)
    label_names = dataset['labels'].unique()
    dataset['labels'] = le.fit_transform(dataset['labels'])
    dataset = pd.get_dummies(dataset, columns=['protocol_type', 'service', 'flag'])
    X = dataset.drop(columns=['labels']).values
    Y = dataset['labels'].values.astype('int')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END, f"{filename} Loaded\nTotal dataset size: {len(dataset)}\n")

def eventVector():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END, f"Data split - Train: {len(X_train)}, Test: {len(X_test)}\n")

class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.softmax(self.fc3(x))
        return x

def neuralNetwork():
    global ann_acc
    y_train_enc = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=len(label_names)).float()
    y_test_enc = torch.nn.functional.one_hot(torch.tensor(y_test), num_classes=len(label_names)).float()
    
    model = ANN(X_train.shape[1], len(label_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, y_train_enc)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        predictions = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1)
    ann_acc = accuracy_score(y_test, predictions.numpy()) * 100
    text.insert(END, f"ANN Accuracy: {ann_acc:.2f}%\n")

def svmClassifier():
    global svm_acc
    model = svm.SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    svm_acc = accuracy_score(y_test, predictions) * 100
    text.insert(END, f"SVM Accuracy: {svm_acc:.2f}%\n")

def graph():
    plt.bar(['ANN', 'SVM'], [ann_acc, svm_acc], color=['green', 'red'])
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.show()

def predictTest():
    global scaler
    test_file = filedialog.askopenfilename()
    if not test_file:
        text.insert(END, "No test file selected.\n")
        return
    test_data = pd.read_csv(test_file)
    for col in ['protocol_type', 'service', 'flag']:
        if col in test_data.columns:
            test_data[col] = test_data[col].astype(str)
    test_data = pd.get_dummies(test_data, columns=['protocol_type', 'service', 'flag'])
    train_columns = set(range(X.shape[1]))
    test_columns = set(test_data.columns)
    missing_cols = train_columns - test_columns
    for col in missing_cols:
        test_data[col] = 0
    test_data = test_data.reindex(columns=sorted(train_columns), fill_value=0)
    try:
        test_features = scaler.transform(test_data.values)
    except Exception as e:
        text.insert(END, f"Error in standardization: {e}\n")
        return
    model = svm.SVC()
    model.fit(X, Y)
    predictions = model.predict(test_features)
    test_data['Predicted_Label'] = le.inverse_transform(predictions)
    text.insert(END, "Prediction Results:\n")
    for i, pred in enumerate(test_data['Predicted_Label'].head(10)):
        text.insert(END, f"Test Sample {i+1}: {pred}\n")
    test_data.to_csv("Predicted_Results.csv", index=False)
    text.insert(END, "Predictions saved to Predicted_Results.csv\n")

text = Text(main, height=20, width=100)
text.place(x=50, y=50)
button_style = {"bg": "lightblue", "fg": "black", "font": ("times", 12, "bold"), "width": 20, "height": 2}
buttons = [
    ("Upload Dataset", upload, 50), ("Generate Event Vector", eventVector, 250),
    ("Neural Networks", neuralNetwork, 450), ("Run SVM", svmClassifier, 650),
    ("Comparison Graph", graph, 850), ("Predict Results", predictTest, 1050)
]
for name, cmd, x in buttons:
    Button(main, text=name, command=cmd, **button_style).place(x=x, y=500)
main.mainloop()