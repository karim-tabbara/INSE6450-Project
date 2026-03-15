# INSE6450-Project

### **Name**: Karim Tabbara
### **Student ID**: 40157871
### **Project Title**: Inbox Triage & Response Helper

---

### Project Folder Structure - Per Milestone
**Milestone 1 (10/02/2026)**
- "data" folder: contains "RawData.csv", which has the raw labeled data
- "outputs" folder: contains the output artifacts
- "src" folder: contains the "main.py" python script --> code for data processing + initial feature engineering

**Milestone 2 (28/02/2026)**
- "data" folder: contains "RawData.csv", which has the raw labeled data
- "outputs" folder: contains the output artifacts
- "models" folder: contains the logistic regression model and the tf-idf vectorizer saved as joblib files
- "src" folder
  - "main.py" --> code to call and execute functions from other python files
  - "preprocessing.py" --> code to preprocess input, load and prepare training data
  - "model.py" --> code to train model, evaluate model, plot learning curve
  - "inference.py" --> code to perform single predictions, measure inference metrics

**Milestone 3 (14/03/2026)**
- "data" folder: contains "RawData.csv", which has the raw labeled data
- "outputs" folder: contains the output artifacts
- "models" folder: contains the logistic regression models and the tf-idf vectorizers saved as joblib files (multiple versions)
  - "logistic_regression_model.joblib" --> model from milestone 2
  - "tfidf_vectorizer.joblib" --> vectorizer from milestone 2
  -  "logistic_regression_model_v2.joblib" --> model from milestone 3
  -  "tfidf_vectorizer_v2.joblib" --> vectorizer from milestone 3
  -  "logistic_regression_model_noisy.joblib" --> model for the adaptation experiment in milestone 3 (part 4, question 3)
  -  "tfidf_vectorizer_noisy.joblib" --> vectorizer for the adaptation experiment in milestone 3 (part 4, question 3)
- "src" folder
  - "main.py" --> code to call and execute functions from other python files
  - "preprocessing.py" --> code to preprocess input, load and prepare training data
  - "model.py" --> code to train model, evaluate model, plot learning curve
  - "inference.py" --> code to perform single predictions, measure inference metrics
  - "adaptation_experiment.py" --> code for the milestone 3 adaptation experiment (part 4, question 3)
  - "monitoring.py" --> code for the milestone 3 monitoring dashboard figure (part 3, question 3)
  - "plots.py" --> code for milestone 3 robustness plots (part 2, question 3)
  - "stress_tests.py" --> code to perform robustness stress tests
---

## Dependencies:
- pandas
- matplotlib
- numpy
- scikit-learn (sklearn)
- scipy
- joblib
- seaborn
- psutil

## Output Artifacts:
All output artifacts are saved in the "outputs" folder

For **Milestone 1**: 
- "labels_distribution.png": label distribution bar plot
- "ProcessedData.csv": CSV containing the labeled processed data (same format as "RawData.csv")

Added For **Milestone 2**:
- "confusion_matrix_Test.png": confusion matrix for test set
- "confusion_matrix_Validation.png": confusion matrix for validation set
- "learning_curve.png": learning curve plots

Added For **Milestone 3**:
- "NoisyData.csv" --> noisy dataset for the milestone 3 adaptation experiment
- "confusion_matrix_Noisy_Adaptation_Experiment.png" --> confusion matrix for the milestone 3 adaptation experiment
- "monitoring_dashboard_simulation.png" --> simulated monitoring dashboard for milestone 3
- "failure_examples.csv" --> some failure examples
- "reliability_diagram.png" --> reliability diagram plot
- "confidence_histogram.png" --> confidence histogram plot
- "robustness_curve.png" --> robustness curve plot
- "confusion_matrix_Masked_Test.png" --> confusion matrix for token masking stress test
- "confusion_matrix_Noisy_Test.png" --> confusion matrix for character noise stress test
- "confusion_matrix_OOD_Test.png" --> confusion matrix for OOD inputs stress test
- "confusion_matrix_Truncated_Test.png" --> confusion matrix for truncated emails stress test

## How To Run:
- Clone the repository
- Make sure all dependencies are installed
- In the terminal, navigate to the "src" folder using "cd src"
- Run one of the python script using "python main.py", "python adaptation_experiment.py", "python monitoring.py", "python stress_tests.py", "python plots.py"
- The output artifacts will be saved in the "outputs" folder + the models will be saved to the "models" folder (overwritten if existing already)
  - Note: the artifacts from Milestone 1, and some artifacts from Milestone 2 are no longer computed as the code has been updated. They are kept in the "outputs" folder.
- For some scripts, information will be printed in the console during execution
