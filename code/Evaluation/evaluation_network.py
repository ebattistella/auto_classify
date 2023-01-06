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


# Prints the feature importances based on SHAP values in an ordered way
# shap_values : The SHAP values calculated from a shap.Explainer object
# features : The name of the features, on the order presented to the explainer
def print_feature_importances_shap_values(shap_values, features):
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
    feature_importances_norm = {k: v for k, v in
                                sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")


def draw(model, x_train, x_test, name):
    shap.initjs()
    # Create object that can calculate shap values
    print(model.predict(x_train))
    if len(x_train) > 50:
        x_train = shap.sample(x_train, 50)
    if len(x_test) > 50:
        x_test = shap.sample(x_test, 50)
    explainer = shap.KernelExplainer(model.predict, x_train)
    print("shap, train", explainer.shap_values(x_train, l1_reg=False))
    # Calculate Shap values
    shap_values = explainer.shap_values(x_test, l1_reg=False)
    print("shap", sum(shap_values))
    np.save(name + "_shap_values", shap_values)
    f = plt.figure()
    shap.force_plot(explainer.expected_value, shap_values, x_test)
    f.savefig(name + "_force.png", bbox_inches='tight', dpi=600)

    f = plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    f.savefig(name + "_bar.png", bbox_inches='tight', dpi=600)

    f = plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type="layered_violin", color='coolwarm')
    f.savefig(name + "_layered_violin.png", bbox_inches='tight', dpi=600)

    f = plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type="violin", color='coolwarm')
    f.savefig(name + "_violin.png", bbox_inches='tight', dpi=600)


def main_function(params):
    if params == -1:
        return
    path, name, x_train, x_test, signature, model = params
    # Feature importance on the train set.
    try:
        draw(model, x_train.iloc[:, signature], x_train.iloc[:, signature], path + "_shap_train_" + name)
    except Exception as e:
        print("train", e)
    # Feature importance on the test set trained on the train set.
    try:
        draw(model, x_train.iloc[:, signature], x_test.iloc[:, signature], path + "_shap_" + name)
    except Exception as e:
        print("test", e)
