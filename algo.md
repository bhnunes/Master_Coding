# Algorithm Specification: Optuna-Optimized Stratified Patient-Level Dataset Split

## 1. Objective
Implement a Python script to split an imbalanced medical imaging dataset of $N$ patients into **Train**, **Validation**, and **Test** sets. The solution must use Bayesian Optimization (via Optuna) coupled with a greedy allocation algorithm to satisfy two strict constraints:
1. **Zero Data Leakage:** Strict patient-level isolation.
2. **Stratification:** The sample-level class distribution in each subset must closely mirror the global sample-level class distribution.

## 2. Dynamic Parameters
The script must accept the following as dynamic inputs (do not hardcode values):
* `dataset`: The raw sample-level DataFrame.
* `n_test_patients`: Integer target capacity for the Test set.
* `n_val_patients`: Integer target capacity for the Validation set.
* The Train set capacity is dynamically calculated as $N - n_{test\_patients} - n_{val\_patients}$.

## 3. Implementation Steps

### Step 1: Patient-Level Aggregation & Profiling
* **Group & Aggregate:** Group the input data by `Patient_ID`. Create a metadata DataFrame with: `Patient_ID`, `Total_Samples`, `Cancer_Samples`, and `Non_Cancer_Samples`.
* **Global Target:** Calculate the `Global_Cancer_Ratio` = (Total Cancer Samples) / (Total Samples).

### Step 2: The Optuna Objective Function
Define an objective function for Optuna to minimize. Inside this function:
1. **Dynamic Priority Assignment:** Iterate through the $N$ unique patients. For each patient, use Optuna to suggest a continuous float between $0.0$ and $1.0$ (`trial.suggest_float(f"weight_{patient_id}", 0.0, 1.0)`).
2. **Sorting:** Sort the patient metadata DataFrame based on these Optuna-suggested weights.
3. **Greedy Allocation:** * Initialize empty Train, Validation, and Test containers with tracking counters (patient count, total samples, cancer samples).
    * Iterate through the sorted patient list.
    * Check which sets still have available patient capacity (e.g., `current_test_patients < n_test_patients`).
    * For the sets with available capacity, calculate the hypothetical new cancer ratio if the current patient's samples were added.
    * Calculate the absolute delta between each hypothetical ratio and the `Global_Cancer_Ratio`.
    * Assign the patient to the available set with the smallest delta.
4. **Loss Calculation:** Once all $N$ patients are placed, calculate the final error score:
   $$Loss = |Ratio_{Train} - Ratio_{Global}| + |Ratio_{Val} - Ratio_{Global}| + |Ratio_{Test} - Ratio_{Global}|$$
5. **Return Loss:** Return this value for Optuna to minimize.

### Step 3: Optimization Execution
* Initialize an Optuna study with a `TPESampler` for deterministic reproducibility (set a random seed).
* Run the optimization for a designated number of trials (e.g., `n_trials=1000`).
* Extract the best trial parameters (the optimal patient sorting weights).
* Run the greedy allocation logic one final time using these optimal weights to generate the final dataset split.

### Step 4: Statistical Verification (Pure NumPy)
* Utilize pure `numpy` to calculate the Chi-square statistic for the final splits.
* For each subset (Train, Val, Test):
  1. Calculate the Expected Cancer Samples: `Expected_Cancer = Total_Set_Samples * Global_Cancer_Ratio`
  2. Calculate the Expected Non-Cancer Samples: `Expected_Non_Cancer = Total_Set_Samples * (1 - Global_Cancer_Ratio)`
  3. Compute the Chi-square statistic using the formula: 
     `chi2_stat = np.sum((Observed - Expected)**2 / Expected)`
* The script must use this raw `chi2_stat` as the metric of verification (a value closer to 0 indicates a better fit).

## 6. Technology Stack
* `pandas` for data manipulation.
* `numpy` for core mathematical and statistical operations (no `scipy` or `sklearn`).
* `optuna` for the TPE Sampler optimization loop.