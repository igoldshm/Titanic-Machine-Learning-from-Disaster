# Titanic-Machine-Learning-from-Disaster
In this project, we developed a predictive model to determine whether a passenger would have survived the Titanic disaster.
## **Dependencies**
To run the project, install the required Python libraries:

```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
```
## Data
To achieve this, we used tabular data from Kaggle, consisting of two files: train_data.csv and test_data.csv. The train.csv file contains information about a subset of passengers who were on board, totaling 891 individuals. Each passenger is represented as a separate row in the dataset. The dataset included detailed information about passengers and their tickets, such as age, gender, class, fare, and other relevant attributes.
The second column, "Survived," indicates whether a passenger survived the disaster:

    A value of 1 means the passenger survived.
    A value of 0 means the passenger did not survive.
We used this column as the ground truth for our training, representing it as "y" in our model.
test_data.csv contains the other 418 passengers on board which we needed to predict if survived.
### Feature engineering
potentially we could have looked at the data and manually try to find patterns for each column/combinations of columns (for example gender as an indicator for survival rate), but that would have taken us a lot of time and it would have been very challenging to complete. Instead we build a random forest model. This model creates several decision trees, where each tree represent a different combinatorial pattern in the data and each tree goes over all the passengers and determine if they survived ("1") or not("0"). The final prediction of the model for each passenger is being determined democratically according to what was the most predicted outcome by all the trees.
## Model
The model was set to be with 100 estimators (trees) that have max_depth=5 (number of layers for each tree). The model looks for patterns in four different columns ("Pclass", "Sex", "SibSp", and "Parch") of the data.

## Results
