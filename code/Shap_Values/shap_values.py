####################################
### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Shapley; Shap Values;
### Description: <Leverage Shap library to  compute and plot the importance of the features>
### Input: a trained model, the data the importance has to be computed on, a template name to use for saving plots
### Ouput: three .png files with the different plots
###################################


import shap  # package used to calculate Shap values
import matplotlib.pyplot as plt

def draw(model, data, name):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)

    # Calculate Shap values
    shap_values = explainer.shap_values(data)

    f = plt.figure()
    shap.summary_plot(shap_values[1], data, plot_type="bar")
    f.savefig(name + "_bar.png", bbox_inches='tight', dpi=600)

    f = plt.figure()
    shap.summary_plot(shap_values[1], data, plot_type="layered_violin", color='coolwarm')
    f.savefig(name + "_layered_violin.png", bbox_inches='tight', dpi=600)

    f = plt.figure()
    shap.summary_plot(shap_values[1], data, plot_type="violin", color='coolwarm')
    f.savefig(name + "_violin.png", bbox_inches='tight', dpi=600)


# Use example
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd

    iris = load_iris()
    x = pd.DataFrame(iris.data[:, :2], columns=["Feature 0", "Feature 1"])
    y = pd.DataFrame(iris.target)
    model = DecisionTreeClassifier()
    model.fit(x, y)

    draw(model, x, "iris_shap")