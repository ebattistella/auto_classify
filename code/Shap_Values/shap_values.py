import shap  # package used to calculate Shap values
import matplotlib.pyplot as plt

def draw(model, data, name):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)

    # Calculate Shap values
    shap_values = explainer.shap_values(data)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data, show=False)
    plt.savefig(name + "_force.png")
    f = plt.figure()
    shap.summary_plot(shap_values, data, plot_type="bar")
    f.savefig(name + "_summary.png", bbox_inches='tight', dpi=600)

    f = plt.figure()
    shap.summary_plot(shap_values, data)
    f.savefig(name + "_summary.png", bbox_inches='tight', dpi=600)

# Use example
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris()
    x = iris.data[:, :2]
    y = iris.target
    model = DecisionTreeClassifier()
    model.fit(x, y)

    draw(model, x, "iris_shap")