Credit Card Fraud Detection with Explainable AI ￼

This project builds and compares several machine‑learning models to detect credit card fraud, and then explains their decisions using XAI (Explainable AI) techniques like SHAP and LIME. It is designed so that even beginners can follow what is happening step by step.

The goal is not only to get good accuracy, but also to understand why the models flag a transaction as fraud, and how reliable and fast these explanations are.

1. What this project does ￼

At a high level, the script:

1. Loads the popular ‎`creditcard.csv` dataset (anonymised credit‑card transactions with a ‎`Class` label: 0 = legitimate, 1 = fraud).

2. Handles the strong class imbalance using SMOTE (oversampling fraud cases in the training set).

3. Trains and evaluates three models:

 ▫ Logistic Regression (baseline)

 ▫ XGBoost (tuned with RandomizedSearchCV)

 ▫ Random Forest (tuned with RandomizedSearchCV)

4. Compares models using:

 ▫ ROC‑AUC, PR‑AUC

 ▫ F1‑score, Precision, Recall

 ▫ Confusion matrices (TP, FP, TN, FN)

5. Optimises the decision threshold (not just 0.5) to reduce false positives while keeping good fraud detection.

6. Uses SHAP to get:

 ▫ Global feature importance (bar and beeswarm plots)

 ▫ Local explanation (waterfall plots) for individual fraud and legitimate examples

7. Uses LIME to explain specific predictions for all three models (fraud and non‑fraud instances).

8. Measures:

 ▫ Stability of explanations (how much LIME/SHAP rankings change between runs / perturbations)

 ▫ Latency (time in milliseconds to generate one explanation for each method and model)

9. Saves plots and CSV tables that you can use directly in a report or dissertation.

2. Requirements ￼

You need Python and the following packages:

- ‎`pandas`, ‎`numpy`

- ‎`scikit-learn`

- ‎`imbalanced-learn`

- ‎`xgboost`

- ‎`shap`

- ‎`lime`

- ‎`matplotlib`, ‎`seaborn`

Install them with:￼

You also need the dataset file ‎`creditcard.csv` in the same folder as the script.

3. How the code is structured ￼

The script runs as a single file, roughly in this order:

1. Imports and setup

 ▫ Imports all libraries, turns off warnings, and configures Matplotlib to save plots to files (no GUI needed).

2. Data loading and preprocessing

 ▫ Reads ‎`creditcard.csv` into a pandas DataFrame.

 ▫ Drops any rows with missing ‎`Class`.

 ▫ Splits into features ‎`X` and target ‎`y`.

 ▫ Train–test split (stratified to keep class ratio).

 ▫ Scales features with ‎`StandardScaler`.

 ▫ Uses SMOTE on the training data to balance fraud vs. legitimate.

3. Helper functions

 ▫ ‎`save_fig(...)`: saves any current Matplotlib figure as a PNG.

 ▫ ‎`plot_confusion_matrix(...)`: draws and saves a confusion matrix heatmap.

 ▫ ‎`evaluate_model(...)`: given predictions and probabilities, returns key metrics in a dictionary.

4. Model training and evaluation

 ▫ Model 1: Logistic Regression (baseline)

 ⁃ Trained on SMOTE‑balanced, scaled features.

 ⁃ Evaluated on the original test set.

 ⁃ Confusion matrix and metrics printed and saved.

 ▫ Model 2: XGBoost (tuned)

 ⁃ Uses ‎`RandomizedSearchCV` with a parameter grid and 3‑fold stratified CV.

 ⁃ Picks the best model based on F1 score.

 ⁃ Evaluates on test set, prints and saves confusion matrix and metrics.

 ▫ Model 3: Random Forest (tuned)

 ⁃ Same tuning process (RandomizedSearchCV).

 ⁃ Evaluated in the same way.A comparison table (‎`model_comparison_default.csv`) is created for the three models at the default threshold 0.5.

5. PR and ROC curves

 ▫ Plots Precision–Recall and ROC curves for all three models on the same figure:

 ⁃ ‎`pr_curves.png`

 ⁃ ‎`roc_curves.png`

6. Threshold optimisation & false‑positive reduction

 ▫ Function ‎`optimise_threshold(...)`:

 ⁃ Uses the precision–recall curve to:

 ▪ Find the F1‑maximising threshold

 ▪ Find a threshold that achieves at least 80% recall

 ⁃ Compares:

 ▪ Default threshold (0.5)

 ▪ F1‑optimal threshold

 ▪ 80% recall threshold

 ⁃ Prints TP, FP, FN, F1, and Recall for each.

 ⁃ Reports percentage reduction in FP from default to F1‑optimal.

 ▫ Runs this for all three models.

 ▫ For Random Forest, it also:

 ⁃ Plots F1 vs threshold and precision/recall vs threshold (‎`threshold_analysis_rf.png`).

 ⁃ Saves confusion matrix at optimal threshold (‎`ConfusionMAT3_optimal.png`).

 ▫ A table of metrics at the optimal thresholds is saved as ‎`model_comparison_optimal.csv`.

7. SHAP analysis

 ▫ Builds SHAP explainers:

 ⁃ ‎`LinearExplainer` for Logistic Regression.

 ⁃ ‎`TreeExplainer` for XGBoost and Random Forest.

 ▫ Takes a sample of 500 test instances.

 ▫ Generates:

 ⁃ Global feature importance bar plots and beeswarm plots for each model:

 ▪ ‎`shap1_bar.png`, ‎`shap1_beeswarm.png`

 ▪ ‎`shap2_bar.png`, ‎`shap2_beeswarm.png`

 ▪ ‎`shap3_bar.png`, ‎`shap3_beeswarm.png`

 ▫ Selects:

 ⁃ One fraud instance

 ⁃ One legitimate instance

 ▫ Creates SHAP waterfall plots (top 10 features) showing how each feature pushes the prediction:

 ⁃ Fraud:

 ▪ ‎`shap1_waterfall_fraud.png` (Logistic Regression)

 ▪ ‎`shap2_waterfall_fraud.png` (XGBoost)

 ▪ ‎`shap3_waterfall_fraud.png` (Random Forest)

 ⁃ Legitimate:

 ▪ ‎`shap3_waterfall_legit.png` (Random Forest)

8. LIME analysis

 ▫ Builds a ‎`LimeTabularExplainer` using the SMOTE‑balanced training data.

 ▫ For all three models, and for both the fraud and legitimate instances, it generates local LIME explanations as bar plots:

 ⁃ Fraud:

 ▪ ‎`lime1_fraud.png`, ‎`lime2_fraud.png`, ‎`lime3_fraud.png`

 ⁃ Legitimate:

 ▪ ‎`lime1_legit.png`, ‎`lime2_legit.png`, ‎`lime3_legit.png`

9. Explanation stability (RQ1)

 ▫ LIME stability:

 ⁃ Runs LIME 10 times on the same fraud instance with different random seeds.

 ⁃ Records how the rank order of important features changes.

 ⁃ Stores mean rank, standard deviation of rank, and how often each feature appears in the top 10.

 ⁃ Saves a table ‎`lime_stability.csv`.

 ⁃ Plots the rank standard deviation for the top features (‎`lime_stability_plot.png`).

 ▫ SHAP stability:

 ⁃ Takes the fraud instance for Random Forest.

 ⁃ Adds small Gaussian noise (σ = 0.05) 10 times.

 ⁃ Recomputes SHAP values and checks how many of the top‑10 features stay the same.

 ⁃ Prints mean and std of this overlap.

 ⁃ Saves summary to ‎`shap_stability_summary.csv`.

10. Latency benchmarking (RQ2)

 ▫ Measures how long (in milliseconds) it takes to produce one explanation:

 ⁃ SHAP (for each of the three models)

 ⁃ LIME (for each of the three models)

 ▫ Repeats each measurement 5 times and averages.

 ▫ Saves results to ‎`latency_benchmark.csv`.

 ▫ Visualises them in a bar chart: ‎`latency_benchmark.png`.

11. Final summary table

 ▫ For each model (at its optimal threshold), collects:

 ⁃ ROC‑AUC, PR‑AUC, F1, Precision, Recall, FP, FN

 ⁃ Optimal threshold

 ⁃ SHAP latency (ms)

 ⁃ LIME latency (ms)

 ▫ Prints and saves this as ‎`dissertation_results_summary.csv`, which is ready to be used as a results table in a dissertation or report.

 ▫ Finally, prints a list of all generated files.

4. How to run ￼

1. Place the script (e.g. ‎`fraud_xai_experiments.py`) and ‎`creditcard.csv` in the same folder.

2. Install the dependencies listed above.

3. Run:

￼

4. When it finishes, check the folder for:

 ▫ PNG images of confusion matrices, PR/ROC curves, SHAP and LIME plots, stability and latency plots.

 ▫ CSV files with model comparisons, stability metrics, latency metrics, and a final dissertation summary.

5. Intended use ￼

This repository is mainly intended for:

- Students and beginners who want a complete, end‑to‑end example of:

 ▫ Fraud detection with imbalanced data

 ▫ Hyperparameter tuning for tree‑based models

 ▫ Threshold optimisation and evaluation metrics

 ▫ Explainable AI with SHAP and LIME

- Researchers who need:

 ▫ Ready‑to‑use plots and tables for a report

 ▫ A reference implementation for stability and latency analysis of explanation methods.

The code is heavily commented and prints clear console output so you can follow each step even if you are new to machine learning and XAI.
