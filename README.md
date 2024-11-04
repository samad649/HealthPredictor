# HealthPredictor
CSCE 470 Final Project

Dependencies:
Used the following python modules:
boto3
io
pandas
sklearn

Access: 
This code is public but to modify and test, you will require the access key and private key to an aws s3 bucket.
the data i used from mimic requires i not share it

Summary:
Currently i picked the info that i thought would be most useful & relevant to prediciting a diagnosis.
As this class is information storage & retrieval, i used the clinic free text notes for each patient to create this naive bayes implementation.
While it may not be the most effective i may incorperate other features to help bost accuracy scores.

Future Plans:
As of now, the accuracys scores vary based on the number of samples used to train and test,
also facing an issue where some the rarer diagnosis not being present in the training set so when they appear i am not able to predict.
Add other features like demographic and lab scores to algorithm to imporve scores.


