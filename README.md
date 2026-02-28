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

## How To Run:
- Clone the repository
- Make sure all dependencies are installed
- In the terminal, navigate to the "src" folder using "cd src"
- Run the python script using "python main.py"
- The output artifacts will be saved in the "outputs" folder + the models will be saved to the "models" folder (overwritten if existing already)
- Some information about label distribution and matrix shapes will be printed in the terminal
