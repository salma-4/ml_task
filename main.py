
import tkinter as tk
from tkinter import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression , Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
from tkinter import filedialog

root = tk.Tk()
root.geometry("1000x1000")
root.title('Welcome')
windowFrame = Frame(width='1600', height='799', bg='#526D82')
windowFrame.place(x=1, y=1)
dataset=""

#load dataset
def load_dataset():
    global dataset
    # Ask the user to select a dataset file
    file_path = filedialog.askopenfilename()
    dataset = pd.read_csv(file_path)

    text1.delete("1.0", "end")
    text1.insert(tk.INSERT, dataset.to_string())

    global x
    global y
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    #     for describe data
    des = dataset.describe()
    des = tk.Label(root, text=des, font=("Arial", 10))
    des.place(x=50, y=550)


# Function to calculate the error rate
def calculate_test_rate():
    # Get the selected test size and error type
    test_size = slider.get()
    alpha_value = alpha_slider.get()
    error_type_value = error_type.get()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=float(test_size), random_state=0)

    # Train the  regression model
    rate = " "

    if (float(alpha_value) == 0.0):
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

    else:
        lasso = Lasso(alpha=float(alpha_value))
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)

    # Calculate the error rate based on the selected error type
    if error_type_value == "meanabs":
        rate = metrics.mean_absolute_error(y_test, y_pred)
    elif error_type_value == "meansquared":
        rate = metrics.mean_squared_error(y_test, y_pred)
    elif error_type_value == "rootmean":
        rate = metrics.mean_squared_error(y_test, y_pred, squared=False)

    tk.Label(root, text=rate).place(x=820, y=500)


def calculate_train_rate():
    # Get the selected test size and error type
    test_size = slider.get()
    alpha_value = alpha_slider.get()
    error_type_value = error_type.get()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)

    # Train the  regression model
    train = " "

    if (float(alpha_value) != 0.0):
        lasso = Lasso(alpha=float(alpha_value))
        lasso.fit(X_train, y_train)
        train_pred = lasso.predict(X_train)
    else:
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        train_pred = regressor.predict(X_train)

        # Calculate the error rate based on the selected error type
    if error_type_value == "meanabs":
        train = metrics.mean_absolute_error(y_train, train_pred)
    elif error_type_value == "meansquared":
        train = metrics.mean_squared_error(y_train, train_pred)
    elif error_type_value == "rootmean":
        train = np.sqrt(metrics.mean_squared_error(y_train, train_pred))

    tk.Label(root, text=train).place(x=820, y=540)


# Create a button to select the dataset file
select_button = tk.Button(root, text="Select Dataset", command=load_dataset, width=14)
select_button.place(x=50, y=30)

frame = tk.Frame(root, width=440, height=340)
frame.place(x=50, y=80)

lab=tk.Label(root,text="Data Description",font=("Arial",16))
lab.place(x=50,y=500)
# Scrollbar
v = tk.Scrollbar(frame, orient='vertical')
v.pack(side=tk.RIGHT, fill='y')

h = tk.Scrollbar(frame, orient='horizontal')
h.pack(side=tk.BOTTOM, fill='x')

text1 = tk.Text(frame, yscrollcommand=v.set, wrap=tk.NONE, xscrollcommand=h.set)
text1.pack()

v.config(command=text1.yview)
h.config(command=text1.xview)

slider_label = tk.Label(root, text="Select Test Size: ",font=("Arial",11))
slider_label.place(x=700, y=290)

# Create the slider bar
slider = tk.Scale(root, from_=0.1, to=0.9, resolution=0.1, orient=tk.HORIZONTAL)
slider.place(x=850, y=280)

# create alpha bar
alpha_label = tk.Label(root, text="Select Alpha Value: ",font=("Arial",11))
alpha_label.place(x=700, y=220)

# Create the alpha slider
alpha_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL)
alpha_slider.place(x=850, y=210)

# Create a radio button for selecting the error type
error_type = tk.StringVar()
error_type.set("meanabs")

mae_rb = tk.Radiobutton(root, text="MAE", variable=error_type, value="meanabs")
mae_rb.place(x=700, y=460)

mse_rb = tk.Radiobutton(root, text="MSE", variable=error_type, value="meansquared")
mse_rb.place(x=700, y=400)

rms_rb = tk.Radiobutton(root, text="RMSE", variable=error_type, value="rootmean")
rms_rb.place(x=700, y=430)
# Create a label for the error type
error_label = tk.Label(root, text="Select Error Type: ",font=("Arial",16))
error_label.place(x=700, y=350)

tk.Button(root,text="Test", command=calculate_test_rate,font=("Arial",13)).place(x=700,y=500)

tk.Button(root,text="Train", command=calculate_train_rate ,font=("Arial",13)).place(x=700,y=540)

##prevent the frame widget from resizing itself
frame.pack_propagate(False)
frame.grid_propagate(False)

# Run the GUI main loop
root.mainloop()
