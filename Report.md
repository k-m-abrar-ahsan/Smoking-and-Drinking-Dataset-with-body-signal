
### Data Analysis Report on Smoking and Drinking Prediction Using Machine Learning

#### 1. Introduction

The ability to predict smoking and drinking behaviors is crucial for advancing public health research, enabling targeted interventions and risk assessments. The dataset under analysis consists of health-related indicators, including demographic information and physiological measures (e.g., age, blood pressure, cholesterol levels), to predict whether an individual smokes and/or drinks. With these insights, public health strategies can be tailored to reduce the risks associated with smoking and drinking.

This report examines the performance of multiple machine learning models—specifically K-Nearest Neighbors (KNN), Random Forest, and other classifiers—in predicting smoking and drinking behavior. The primary goal of this analysis was to assess the predictive accuracy and interpretability of the models, with a focus on identifying strategies to improve prediction accuracy, particularly for imbalanced class distributions.

#### 2. Dataset Description

The dataset used for this analysis contains health and demographic data on individuals, specifically designed to capture various physiological and behavioral characteristics. It was collected from kaggle (Link: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data) The dataset comprises 991,346 rows and 30 columns, representing individuals' attributes and health indicators. For the sake of model simplicity and computational efficiency, the dataset was reduced prior to analysis.

##### Key Features of the Dataset:

- Demographic Features:
  - sex: Gender of the individual (Male/Female)
  - age: Age of the individual in years
  - height: Height of the individual in centimeters
  - weight: Weight of the individual in kilograms
  - waistline: Waistline measurement in centimeters

- Health Indicators:
  - sight_left, sight_right: Visual acuity (scores for the left and right eye)
  - hear_left, hear_right: Hearing capability (binary values indicating hearing ability)
  - SBP: Systolic blood pressure (mmHg)
  - DBP: Diastolic blood pressure (mmHg)
  - BLDS: Blood sugar levels (measured in mg/dL)
  - tot_chole, HDL_chole, LDL_chole: Total cholesterol, HDL cholesterol (good), and LDL cholesterol (bad) levels (measured in mg/dL)
  - triglyceride: Triglyceride levels (mg/dL)
  - hemoglobin: Hemoglobin levels (g/dL)
  - serum_creatinine: Serum creatinine levels (mg/dL)
  - SGOT_AST, SGOT_ALT: Enzyme levels related to liver function (measured in U/L)
  - gamma_GTP: Gamma-glutamyl transferase (GGT) levels (U/L)

- Lifestyle Indicators:
  - SMK_stat_type_cd: A categorical variable indicating the smoking status (e.g., smoker, non-smoker)
  - DRK_YN: A categorical variable indicating the drinking status (e.g., drinker, non-drinker)
  - urine_protein: Protein presence in urine (indicative of kidney function)

The target variables of interest in this analysis are:
- SMK_stat_type_cd: Smoking status (whether the individual is a smoker or not).
- DRK_YN: Drinking status (whether the individual is a drinker or not).

##### Data Preprocessing:

Prior to training the models, several preprocessing steps were performed on the dataset to ensure the quality of the input data:
1. Handling Missing Data: Missing values were imputed using mean imputation for continuous variables (e.g., height, weight, blood pressure) and mode imputation for categorical variables (e.g., smoking status, drinking status).
   
2. Feature Scaling: Given the presence of features with varying units and scales (e.g., age vs. cholesterol levels), standardization was applied to ensure that no single feature dominated the learning process due to its scale. This transformation centered the features around zero with a unit variance, ensuring that each model could effectively process the data.

3. Class Imbalance: The target variable (smoking or drinking behavior) exhibited class imbalance, with the majority class (non-smokers and non-drinkers) significantly outnumbering the minority class. Techniques like SMOTE (Synthetic Minority Over-sampling Technique) were employed to synthetically balance the dataset, ensuring that the models were trained on an evenly distributed class representation.

4. Feature Selection: Initial exploratory data analysis (EDA) revealed that certain features, including age, height, waistline, systolic/diastolic blood pressure (SBP/DBP), and cholesterol levels (HDL, LDL), exhibited a higher correlation with smoking and drinking behaviors. These features were retained, while irrelevant or highly correlated variables were excluded through a combination of domain expertise and statistical measures (e.g., correlation matrices and p-values).

#### 3. Exploratory Data Analysis (EDA)

Preliminary analysis revealed notable relationships between various health indicators and smoking/drinking behavior. For example:

- Age and Weight: Older individuals tended to have higher cholesterol levels and more pronounced health issues (e.g., high blood pressure), which correlated with higher drinking and smoking rates.
- Blood Pressure: Both systolic and diastolic blood pressure levels exhibited significant differences between smokers/non-smokers and drinkers/non-drinkers.
- Cholesterol Levels: Elevated levels of LDL cholesterol were observed in smokers, indicating that smoking may contribute to higher cardiovascular risk.

These insights formed the basis for selecting the most relevant features for model training.

#### 4. Model Evaluation and Results

##### 4.1 K-Nearest Neighbors (KNN) Classifier

The K-Nearest Neighbors (KNN) model was trained on the preprocessed dataset and evaluated based on accuracy, precision, recall, and F1-score. The model's performance on smoking and drinking predictions is summarized below.

Drinking Prediction:
- Accuracy: 70.59%
- Precision, Recall, F1-Score: KNN's performance was reasonable, with a balanced precision-recall trade-off. However, the model was slightly conservative in predicting drinkers (i.e., a tendency to classify more individuals as non-drinkers).

Smoking Prediction:
- Accuracy: 70.59%
- Precision, Recall, F1-Score: Similarly, KNN performed consistently on smoking predictions, though it struggled with distinguishing between smokers and non-smokers, likely due to the high dimensionality of the feature space and the influence of outliers.

Interpretation:
KNN, being a distance-based algorithm, can be sensitive to the curse of dimensionality, particularly in high-dimensional feature spaces. The standardized features mitigated some of these issues, but the method still struggled with class imbalance and the intricacies of feature interactions.

Suggestions for Improvement:
- Hyperparameter Tuning: The value of k (number of neighbors) is crucial for KNN's performance. A more systematic approach, such as grid search or cross-validation, should be employed to find the optimal value for k.
- Feature Engineering: To improve model performance, further feature selection using techniques like Recursive Feature Elimination (RFE) could be employed to retain only the most influential features, thereby reducing the dimensionality and improving the algorithm’s efficiency.

##### 4.2 Random Forest Classifier

Random Forest, an ensemble learning method, combines multiple decision trees to reduce variance and overfitting, making it particularly suitable for high-dimensional and noisy datasets.

Drinking Prediction:
- Accuracy: 72.10%
- Precision, Recall, F1-Score: Random Forest demonstrated a stronger performance in predicting drinking behavior. The precision and recall were balanced, with the model managing to differentiate effectively between drinkers and non-drinkers.

Smoking Prediction:
- Accuracy: 79.49%
- Precision, Recall, F1-Score: The Random Forest classifier achieved excellent accuracy in smoking prediction. Precision for non-smokers was higher, but recall for smokers was lower, indicating the model’s tendency to predict the majority class more accurately.

Interpretation:
Random Forest’s superior performance can be attributed to its ability to model non-linear relationships and interactions between features. However, the class imbalance still impacted its ability to accurately predict the minority class (smokers), as evident in the lower recall scores for smokers.

Suggestions for Improvement:
- Class Weight Adjustment: Random Forest offers an option to adjust class weights, which could help address the imbalance by assigning higher importance to the minority class (smokers and drinkers).
- Ensemble Methods: Incorporating other ensemble methods such as Gradient Boosting Machines (GBM) or XGBoost could further improve predictive accuracy by leveraging boosting techniques to correct errors made by earlier trees.
- Hyperparameter Tuning: Fine-tuning hyperparameters like the number of trees (n_estimators), maximum depth (max_depth), and minimum sample splits could further optimize the model's performance.

#### 5. Model Comparison and Insights

- KNN vs. Random Forest: While KNN performed reasonably well, it was outperformed by Random Forest in both accuracy and robustness. The ability of Random Forest to capture feature interactions made it the superior model, especially for complex datasets like this one. KNN, on the other hand, showed some limitations in handling high-dimensional data and imbalanced classes.
  
- Class Imbalance: Both models showed a tendency to favor the majority class, which is common in imbalanced datasets. Random Forest demonstrated a better capacity to balance precision and recall, but the minority class (smoking behavior) still suffered in terms of recall. This indicates that the models, though effective, would benefit significantly from techniques designed specifically to handle imbalanced datasets.

- Feature Importance: Random Forest's ability to provide feature importance is particularly useful for understanding which features contribute most to the prediction. Insights into these features could guide further research and interventions in public health by identifying which factors most strongly influence smoking and drinking behavior.

#### 6. Conclusion

In this analysis, we assessed the effectiveness of machine learning models in predicting smoking and drinking behaviors using a comprehensive health dataset. The Random Forest classifier proved to be the most effective model, outperforming KNN in terms of accuracy, robustness, and overall performance. However, the results indicate that both models could be improved by addressing the class imbalance and fine-tuning hyperparameters.

Further steps include:
- Implementing more advanced ensemble methods such as Gradient Boosting Machines or XGBoost to improve predictive performance.
- Exploring the use of SMOTE or cost-sensitive learning approaches to tackle class imbalance more effectively.
- Conducting hyperparameter optimization via cross-validation to further refine both models.

This study demonstrates the utility of machine learning for public health predictions and highlights several avenues for future research and model enhancement.
