### Author: Enzo Battistella
### Date: 3/17/2022
####################################
### Keywords: Classification; Feature Selection; Models Tuning; Model Selection;
### Description: <Leverage the feature selection and classification frameworks to provide a full end to end pipeline>
### Input: data, ground truth, learned distance
### Ouput: signature of features, models, predictions, assessment of the classification
###################################
import pickle

from ..Classification import classification
import pickle


def main_function(data, target, model_name, features_name):
    features = np.load(name_classif, allow_pickle=True)
    model = pickle.load(open(name + ".sav", "rb"))
    y_pred = model.predict(data.iloc[:, features])
    classification.assessment(target, y_pred)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    model_name = ""
    features_name = ""
    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
    print(main_function(data, target, model_name, features_name))
