"""Sample code for implementing PCA to reduce the dimensionality of your feature
 matrix (simple example)"""


"""Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score , classification_report
from sklearn.decomposition import KernelPCA

"""# Import the penguin dataset"""
iris = sns.load_dataset("iris")

"""#Typical queries used to see what your data looks like - always carry this out before completing any analysis
    on your data"""
iris.head()
iris.info()
iris.describe()
iris.columns
iris.isnull().sum() # thera are no null values in the data

"""Check the correlation between each of the vars"""
"""Sepal length and Sepal width as well as petal length and petal with look to be highly correlated"""
sns.pairplot(iris)
iris.corr()

################################################################################################################
                # Support vector machine used to predict the species
################################################################################################################

"""# Importing the dataset"""
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

"""# Splitting the dataset into the Training set and Test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

"""# Feature Scaling"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

"""# Training the SVM model on the Training set
# Set the kernel = linear"""
"""Please note one should conduct hyperparameter tuning in order to determine the most appropriate 
   hyperparameters to use"""
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""
y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""
"""Overall accuracy is 97% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 97% 
   There was only on case which was misclassified"""
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy_score(y_test, y_pred)


##############################################################################################################
                                        # PCA
##############################################################################################################

"""The problem we may have is that we may have far too many features.  PCA is used to
   represent a multivariate data table as smaller set of variables (summary indices) in
   order to observe trends, jumps, clusters and outliers
"""

"""We will look to decrease the number of features down to just two"""


"""Applying Kernel PCA
   # Use radial basis function (similar to the normed distance) from the landmarks"""
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

"""How much variance is retained?"""
kpca_transform = kpca.fit_transform(X_train)
explained_variance = np.var(kpca_transform, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
np.cumsum(explained_variance_ratio)


"""# Training the SVM model on the Training set
# Set the kernel = linear"""
"""Please note one should conduct hyperparameter tuning in order to determine the most appropriate 
   hyperparameters to use"""
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


"""# Predicting the Test set results"""
y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""
"""Overall accuracy is 84% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 83% 
   There are now 6 cases misclassified"""
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy_score(y_test, y_pred)



"""Visualize the data"""
results_df = pd.concat([pd.DataFrame(X_test,columns=['feature_1','feature_2']),
                        pd.DataFrame(y_pred,columns=['target_pred']),
                        pd.DataFrame(y_test, columns=['target'])],axis=1)

results_df.loc[results_df['target_pred'].str.strip() != results_df['target'].str.strip(),'misclassified']='misclassified'
results_df.fillna('correct',inplace=True)

results_df['est_category'] = results_df['target'].astype(str) + '_' + results_df['misclassified'].astype(str)
print(results_df['est_category'])

#plt.clf()
sns.set()
_ = sns.scatterplot(data=results_df, x='feature_1', y='feature_2', hue= 'target')
_ = plt.xlabel('Feature 1')
_ = plt.ylabel('Feature 2')
_ = plt.title('Feature Comparison', fontsize=20)
plt.plot()



# Visualize the misclassified predictions.
plt.clf()
# sns.set()
_ = sns.scatterplot(data=results_df, x='feature_1', y='feature_2', hue= 'est_category', palette='deep')
_ = plt.xlabel('Feature 1')
_ = plt.ylabel('Feature 2')
_ = plt.title('Feature Comparison', fontsize=20)
plt.plot()

