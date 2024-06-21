

---

# Credit Score Evaluation

This project involves evaluating credit scores using machine learning models. The notebook includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Requirements

Before running the code, ensure you have the following installed:

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Python Packages

You can install the required packages using `pip`. Below is the list of packages used in this project:

- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`
- `scikit-learn`
- `datasist`
- `imbalanced-learn`
- `category_encoders`
- `xgboost`
- `joblib`

To install all the dependencies at once, you can use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the packages:

```bash
pip install matplotlib seaborn pandas numpy scikit-learn datasist imbalanced-learn category_encoders xgboost joblib
```

## Usage

1. Clone the repository:

```bash
git clone
https://github.com/Tvishak16/Customer-Classification-on-Credit-Score.git

cd Customer-Classification-on-Credit-Score
```



2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open the `creditscoreevaluation.ipynb` notebook and run the cells to execute the code.

## Project Structure

- `creditscoreevaluation.ipynb`: The main Jupyter notebook containing all the code for data loading, preprocessing, EDA, and model training.
- `train.csv`: Training dataset.
- `test.csv`: Test dataset.

## Data Preprocessing

The notebook includes steps for data preprocessing such as:

- Loading datasets
- Handling missing values
- Encoding categorical variables
- Detecting and handling outliers
- Splitting data into training and test sets
- Applying transformations

## Model Training and Evaluation

The notebook uses various machine learning models for training and evaluation, including:

- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Random Forest
- Extra Trees
- Bagging Classifier
- Stacking Classifier
- AdaBoost
- XGBoost

The models are evaluated using metrics like confusion matrix and classification report.

## Visualization

The project utilizes `matplotlib` and `seaborn` for visualizing the data distributions, feature importance, and model performance.

## Contributing

Feel free to contribute to this project by submitting a pull request or opening an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License.

---

