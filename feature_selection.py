import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import smtplib

# Load the NSL-KDD dataset
df = pd.read_csv('path_to_nsl_kdd_dataset.csv')

# Select features for training
selected_features = ['duration', 'src_bytes',
'dst_bytes', 'wrong_fragment', 'urgent',
'hot', 'num_failed_logins', 'logged_in',
'num_compromised',
'root_shell', 'su_attempted',
'num_root', 'num_file_creations',
'num_shells', 'num_access_files',
'num_outbound_cmds',
'is_host_login', 'is_guest_login',
'count', 'srv_count',
'serror_rate', 'srv_serror_rate',
'rerror_rate', 'srv_rerror_rate',
'same_srv_rate', 'diff_srv_rate',
'srv_diff_host_rate',
'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df['target'], test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Generate classification report
print(classification_report(y_test, y_pred))

# Check for anomalies in user behavior
anomaly_indices = np.where(y_pred == 'anomaly')[0]
if len(anomaly_indices) > 0:
    # Send email notification if anomalies are detected
    sender_email = 'your_email@example.com'
    receiver_email = 'recipient_email@example.com'
    password = 'your_email_password'

    message = "Subject: Anomalies Detected in User Behavior. Anomalies have been detected in the user behavior based on the intrusion detection system."

    Anomaly indices: {}

    "".format(anomaly_indices)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

    print("Anomalies detected. Email notification sent.")
else:
    print("No anomalies detected.")