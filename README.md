PCA (Principal component analysis)
---

Python code for reducing the dimensionality of your feature matrix.  

***About the Python file:***  
The code uses the Iris dataset (on Seaborn) and assesses the model impact of using SVM (Support Vector Machines)
to predict the iris species based on 4 features ('sepal_length', 'sepal_width', 'petal_length' and 'petal_width').

We then decrease the dimensiolatily of the features from 4 to 2 (using Kernel PCA) and assess how it impacts the 
predictive ability of the model. 

Please note the use of PCA in this example is not a good use case. We would tend to use PCA when we have a large number of 
features with a large number of these being non-prdictive and/or highly correlated with each other.

Further information on PCA:
Andrew Ng - https://www.youtube.com/watch?v=T-B8muDvzu0  
Visualize PCA - https://setosa.io/ev/principal-component-analysis/


