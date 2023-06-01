import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import smtplib

# the NSL-KDD dataset
dataset = pd.read_csv('KDDTrain+.csv')

# Preprocessing the dataset
dataset = dataset.dropna()  # Remove any rows with missing values

# Split features and labels
X = dataset.iloc[:, -1] # Features
y = dataset.iloc[:, -1]  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Predict labels for the test set
y_pred = rf_model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Read user behavior data
user_data = pd.read_csv('KDDTest+.csv')

# Predict labels for the user behavior data
user_predictions = rf_model.predict(user_data)

# Check for anomalies
anomalies = user_data[user_predictions == 'anomaly']

# Send email notification if there are any anomalies
if len(anomalies) > 0:
    sender_email = 'your_email@gmail.com' #Sender's email address
    receiver_email = 'recipient_email@gmail.com'  #Recipient's email address
    password = 'your_password' # Sender's email password
    subject = 'Anomalies Detected in User Behavior'
    body = 'Anomalies have been detected in the user behavior.\n\n' + str(anomalies)
    message= 'Subject: {}\n\n{}'.format(subject, body)
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    print('Email notification sent.')

else:
    print('No anomalies detected.')

#KDDTrain+.ARFF: The full NSL-KDD train set with binary labels in ARFF format
#KDDTrain+.TXT: The full NSL-KDD train set including attack-type labels and difficulty