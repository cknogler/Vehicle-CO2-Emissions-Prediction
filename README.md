# Vehicle-CO2-Emissions-Prediction
This project analyzes the ADEME Car Labelling dataset to identify factors influencing vehicle CO₂ emissions in France. It covers data cleaning, EDA, clustering, and predictive modeling, exploring links between emissions and vehicle features while using ML to predict CO₂ and highlight key drivers.
# CO2 Emissions Analysis: Understanding and Optimizing Vehicle Fleet Performance

## Project Overview
This project systematically analyzes vehicle CO2 emissions data to understand the impact of various vehicle characteristics and to derive actionable strategies for emission reduction. The methodology encompasses data preprocessing, exploratory data analysis, advanced clustering for segmentation, and predictive modeling with interpretability analysis.

## Central Research Question
**"How do vehicle characteristics impact CO2 emissions, and how can we optimize for reduction?"**

This guiding question ensures that every analytical step contributes to understanding the drivers of CO2 emissions and informing data-driven strategies for a greener vehicle fleet.

### Analysis Process Flow
Below is a visual representation of the systematic approach taken in this analysis:

<img width="1024" height="1536" alt="Process" src="https://github.com/user-attachments/assets/cb232be8-2926-4750-bdcf-b73d8b290735" />



## Data Preprocessing
Our initial dataset, `cl_JUIN_2013-complet3.csv`, underwent several critical preprocessing steps to ensure data quality and suitability for analysis:

### 1. Column Renaming and Modality Cleaning
Column names were standardized for clarity and consistency (e.g., "Marque" to "Brand", "CO2 (g/km)" to "CO2 (g/km)"). Additionally, specific modalities within categorical columns like 'Gearbox' and 'Field V9' were corrected to unify similar entries.

### 2. Handling Missing Values
Missing values were thoroughly investigated. A heatmap and bar plot visually represented the distribution and count of missing data across columns. 


Key imputation strategies included:
*   **Pollutant Calculation**: Missing `HC (g/km)` and `NOX (g/km)` values were derived from `HC+NOX (g/km)` where possible.
*   **Electric Vehicles**: For electric vehicles (`Fuel == 'EL'`), all pollutant and consumption-related `NaN` values were replaced with `0`, reflecting their zero-emission nature.

### 3. Calculating Average Mass
To create a single representative mass feature, the `Empty Mass Euro Min (kg)` and `Empty Mass Euro Max (kg)` columns were merged into a new column, `Empty Mass Euro Avg (kg)`, which represents the average of the minimum and maximum empty masses. The original min/max columns were subsequently dropped.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset's characteristics, identify patterns, and uncover initial relationships between variables.

### Data Summary and Distributions
Comprehensive `df.describe()` and `df.info()` outputs provided statistical summaries for numerical features and insights into data types and non-null counts. Unique value counts helped in identifying high-cardinality categorical features. Categorical distributions for 'Brand', 'Fuel', and 'Body' revealed the predominant vehicle types and brands in the dataset.

### Fleet-wide Distribution and Frequency Analysis
Visualizations provided a fleet-wide perspective on key attributes:
*   **Technical Distributions**: Histograms showed the distribution of `Empty Mass Euro Avg (kg)`, `Maximum Power (kW)`, `Combined Consumption (l/100km)`, and `CO2 (g/km)`. These often exhibited right-skewness, typical for such vehicle performance metrics.
*   **Categorical Frequencies**: Bar plots illustrated the frequency of different 'Fuel' types, 'Body' types, 'Gearbox' types, and 'Range' categories. A significant proportion of the fleet consisted of diesel (GO) vehicles, particularly minibuses.
*   **Market Comparison**: Bar plots highlighted the top brands and commercial designations, showing Mercedes-Benz as a dominant brand and specific models like VIANO and COMBI having high frequencies.

<img width="2000" height="3000" alt="eda_frequency_distribution_analysis" src="https://github.com/user-attachments/assets/9eeebedd-5169-4b77-bf08-01e02397688f" />


### Primary Drivers of CO2 Emissions
Initial analysis focused on identifying the strongest predictors of CO2 emissions:
*   **Technical Factors**: Scatter plots revealed strong positive correlations between CO2 emissions and `Empty Mass Euro Avg (kg)`, `Maximum Power (kW)`, and `Combined Consumption (l/100km)`. Combined consumption showed the most direct linear relationship.
*   **Categorical Factors**: Box plots demonstrated how CO2 emissions varied significantly across different 'Fuel' types (e.g., diesel vs. petrol), 'Body' types, and 'Gearbox' configurations. These plots showed distinct CO2 distributions for each category, suggesting their importance as drivers.

<img width="2000" height="2800" alt="eda_primary_drivers_co2" src="https://github.com/user-attachments/assets/43d464b5-7124-430f-8d01-e55e71238a9e" />

### CO2 Analysis before Data Reduction
A detailed look at the `CO2 (g/km)` target variable before deduplication showed its distribution, box plot, and Q-Q plot for normality. The distribution was generally right-skewed, indicating more vehicles with lower emissions and a tail of higher-emission vehicles.

<img width="1600" height="800" alt="co2_analysis_before_deduplication" src="https://github.com/user-attachments/assets/2e2b409e-d58a-46d3-9885-8b3e7ec31997" />


### Correlation and Multicollinearity Analysis
Pearson and Spearman correlation heatmaps were generated for all numeric features to assess linear and monotonic relationships, respectively. This helped in understanding inter-feature dependencies.

<img width="2000" height="1000" alt="pearson_vs Spearman_correlation-2" src="https://github.com/user-attachments/assets/c698a2f4-6b57-433c-aa67-18e40629f613" />


Multicollinearity between `Maximum Power (kW)` and `Empty Mass Euro Avg (kg)` was specifically checked using Variance Inflation Factor (VIF). The low VIF values (close to 1) and a near-zero correlation coefficient indicated no strong multicollinearity between these two key features.

### Statistical Hypothesis Testing
*   **Mann-Whitney U Test**: A statistically significant difference (p-value < 0.05) was found between the CO2 emissions of Petrol (ES) and Diesel (GO) vehicles, confirming that fuel type significantly impacts emissions.
*   **Pearson & Spearman Correlations**: Both `Maximum Power (kW)` and `Empty Mass Euro Avg (kg)` showed statistically significant linear and monotonic relationships with `CO2 (g/km)` (p-values < 0.05), affirming their roles as important drivers.

## Data Deduplication and Unique Configuration Identification

To focus on unique mechanical configurations within each Brand and Model and reduce redundancy, especially within the dominant Mercedes-Benz fleet, a deduplication process was applied. This involved several steps:

### 1. Fuel Filter
First, the dataset was filtered to include only Petrol (ES) and Diesel (GO) vehicles, as these represent the vast majority of the fleet and are the primary focus for emission analysis. This step removed non-combustion and niche fuel types.

### 2. Identifying Unique Configurations
Unique vehicle configurations were identified based on a combination of key mechanical and performance attributes:
*   `Brand`, `Folder Model`, `Fuel`, `Body`, `Gearbox`
*   `Maximum Power (kW)`, `Empty Mass Euro Avg (kg)`
*   `CO2 (g/km)`, `Combined Consumption (l/100km)`

Records sharing identical values across these columns were considered duplicates, representing the same mechanical design replicated multiple times in the raw data.

<img width="1600" height="1000" alt="data_deduplication_infographic" src="https://github.com/user-attachments/assets/e44c5468-1851-4001-b174-f175bae24215" />


After filtering for Petrol and Diesel vehicles, an impressive **87.0%** of the remaining records were identified as duplicates of existing mechanical configurations. This process reduced over 43,000 combustion vehicles to just **5,700** unique entries, providing a cleaner and more representative dataset (`df_unique`) for further analysis.

### CO2 Analysis after Deduplication
Following deduplication, the distribution of `CO2 (g/km)` was re-examined. The deduplicated dataset provided a more accurate representation of the diversity of CO2 emissions across distinct vehicle types rather than inflated counts due to redundant data entries.

<img width="1600" height="800" alt="co2_analysis_after_deduplication" src="https://github.com/user-attachments/assets/6fd22128-a905-4169-87ed-9eebe90ac536" />

## Outlier Analysis

Outlier detection was performed on key numerical features in the deduplicated dataset (`df_unique`) using the Interquartile Range (IQR) method to identify unusually high or low values that might disproportionately influence modeling.

### CO2 Outliers
Using the IQR method, **128** entries were identified as outliers for `CO2 (g/km)`. All detected outliers were on the higher end, indicating vehicles with significantly higher CO2 emissions compared to the majority of the fleet. For instance, high-performance sports cars like the Lexus LFA and powerful Mercedes-Benz models were consistently flagged.

### Maximum Power (kW) Outliers
Similarly, **477** entries were identified as outliers for `Maximum Power (kW)`. All of these were high-power vehicles, showing a distribution skewed towards higher power outputs, which is expected given the performance-oriented models present in the dataset.

### Empty Mass Euro Avg (kg) Outliers
Only **1** outlier was detected for `Empty Mass Euro Avg (kg)`. This outlier represented a vehicle with a significantly higher mass, likely a specialized or exceptionally heavy model.

## Relationships after Deduplication

After ensuring data quality and reducing redundancy, a deeper dive into the relationships between key features and CO2 emissions was conducted using the deduplicated dataset (`df_unique`).

### Empty Mass and CO2
A strong positive relationship was observed between vehicle empty mass and CO₂ emissions. The Pearson correlation coefficient **0.69** and Spearman rank correlation **0.65** both indicate a strong association, suggesting that heavier vehicles consistently emit more CO₂. The R² value of 0.46 implies that approximately 46% of the variance in CO₂ emissions can be explained by vehicle mass, with the remaining variation attributable to other factors.

<img width="1600" height="800" alt="mass_co2_correlation" src="https://github.com/user-attachments/assets/7fcaa40b-9e38-44b1-8f38-2b83a7d48077" />


### Combined Consumption and CO2
Combined fuel consumption demonstrated an extremely strong positive linear and monotonic relationship with CO2 emissions. Correlation coefficients were approximately **0.98** (Spearman) and **0.96** (Pearson), with an R² value of **0.96**. This high correlation confirms that `Combined Consumption (l/100km)` is the primary driver of `CO2 (g/km)`.

<img width="1600" height="800" alt="consumption_co2_correlation" src="https://github.com/user-attachments/assets/7c930808-9802-4b59-9ae3-5fca33971d45" />



### Maximum Power and CO2
Maximum power shows a positive relationship with CO₂ emissions, with a moderate Pearson correlation (0.36) and a weaker Spearman correlation (0.18), suggesting a limited and potentially non-linear association. The R² value of **0.45** suggests that approximately 45% of the variance in CO2 emissions is explained by maximum power. While significant, its explanatory power is slightly less than that of empty mass and considerably less than combined consumption.

<img width="1600" height="800" alt="power_co2_correlation" src="https://github.com/user-attachments/assets/46de3c52-018b-4c97-a314-82a5c412b682" />




## Clustering (K-Prototypes) for Segmentation

To identify distinct segments within the vehicle fleet based on their characteristics and CO2 emissions, K-Prototypes clustering was employed. This algorithm is suitable for datasets containing a mix of numerical and categorical features.

### 1. Data Preparation for Clustering
The deduplicated dataset (`df_unique`) was used. Key features for clustering included:
*   **Categorical**: `Body`, `Fuel`, `Gearbox`
*   **Numerical**: `Maximum Power (kW)`, `Empty Mass Euro Avg (kg)`

These features were selected as they represent core vehicle characteristics that influence emissions. Numerical features were scaled using `StandardScaler` to ensure they contributed equally to the distance calculation in the clustering algorithm.

### 2. Elbow Method for Optimal K
The Elbow Method was used to determine the optimal number of clusters (k). The 'cost' (a measure of within-cluster dispersion) was plotted against the number of clusters. The \"elbow point\", where the rate of decrease in cost significantly slows down, suggested an optimal `k`.

<img width="2351" height="1454" alt="kprototypes_elbow_method" src="https://github.com/user-attachments/assets/59edbfb4-b65f-4ee3-adc5-49dced5006dd" />


### 3. Final K-Prototypes Model and Cluster Characteristics
Based on the Elbow Method, **4** clusters were chosen. The resulting clusters exhibit distinct profiles across both numerical and categorical features:

#### Cluster Dashboard
This dashboard provides an overview of the cluster sizes, CO2 distribution, and the relationship between power and mass within each cluster. It highlights how different clusters occupy different regions in the feature space and have varying average CO2 emissions.

<img width="4001" height="2157" alt="cluster_dashboard" src="https://github.com/user-attachments/assets/7f52c629-75de-4037-b6af-fdbc234ae545" />



#### Categorical Distribution per Cluster
Stacked bar charts illustrate the distribution of categorical features (`Body`, `Fuel`, `Gearbox`) within each cluster. These plots reveal the dominant types of vehicles that constitute each segment.

<img width="4804" height="1754" alt="cluster_categorical_distribution" src="https://github.com/user-attachments/assets/12882a2e-7d15-429f-841d-f7a38aa28d77" />



#### Normalized Cluster Profiles (Radar and Heatmap)
To visualize the relative differences between clusters across the key profiling features (`Power`, `Mass`, `CO2`), normalized radar charts and heatmaps were generated. This helps in understanding the unique \"signature\" of each cluster. For example:
*   **Cluster 0**: Might represent entry-level vehicles with lower power, mass, and CO2.
*   **Cluster 1**: Could be medium-sized vehicles with moderate power, mass, and CO2.
*   **Cluster 2**: Potentially compact, efficient vehicles.
*   **Cluster 3**: Likely high-performance or heavy-duty vehicles with higher power, mass, and CO2.

<img width="4834" height="2495" alt="cluster_profiles_normalized_data" src="https://github.com/user-attachments/assets/d721649f-8c42-4c09-8b70-f1ef31433b50" />

These distinct clusters provide valuable segments for targeted emission 

## Predictive Modeling

To predict CO2 emissions based on vehicle characteristics, various regression models were trained and evaluated on the deduplicated dataset (`df_unique`).

### 1. Data Preparation for Modeling
The `df_unique` dataset was split into training and testing sets. Categorical features were one-hot encoded, and numerical features were scaled using `StandardScaler` for linear models. For tree-based models, numerical features were passed through without scaling.

### 2. Feature Set Comparison
Different combinations of features were tested using a Random Forest Regressor and 5-fold cross-validation to identify the optimal set for predicting CO2 emissions. The goal was to find a balance between model complexity and predictive performance (measured by Mean Absolute Error).

<img width="1000" height="500" alt="feature_set_comparison" src="https://github.com/user-attachments/assets/d4a25590-274b-4b28-bcad-da38ba5433d7" />


The comparison revealed that using `all_features` (Empty Mass Euro Avg (kg), Maximum Power (kW), Fuel, Gearbox, Body) yielded the best performance, indicating that all these characteristics contribute to CO2 emission prediction.

### 3. Model Performance
Several regression models, including Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting, were trained and evaluated using 'all features'. Performance was assessed based on R² (coefficient of determination) and Mean Absolute Error (MAE).


<img width="1583" height="884" alt="model_performance-2" src="https://github.com/user-attachments/assets/da353566-4844-4305-8cb1-26e58737d7e0" />



*   **Random Forest** consistently showed the highest R² (around 0.94) and lowest MAE, indicating its superior predictive power and ability to capture complex non-linear relationships in the data.
*   **Gradient Boosting** also performed very well, closely trailing Random Forest.
*   **Linear models** (Linear Regression, Ridge, Lasso) performed reasonably but were outperformed by tree-based models, suggesting non-linear relationships are prominent in the data.

### 4. Random Forest Feature Importance
The Random Forest model's inherent ability to quantify feature importance was leveraged to understand which vehicle characteristics are most influential in predicting CO2 emissions. The importances are permuted through the best feature set: `all_features`.

<img width="1584" height="884" alt="feature_importance-2" src="https://github.com/user-attachments/assets/38e79ce2-cb4b-4f25-800f-407ee583148c" />


The top features driving CO2 emissions were `Empty Mass Euro Avg (kg)` and `Maximum Power (kW)`, followed by `Fuel` type and `Gearbox` type. This aligns with findings from the EDA and correlation analysis, reinforcing the robustness of these drivers.

### 5. Partial Dependence Plots (PDPs)
Partial Dependence Plots illustrate the marginal effect of one or two features on the predicted CO2 emissions, after accounting for the average effect of all other features. They provide intuitive insights into the relationship between predictors and the target variable.

<img width="1584" height="884" alt="partial_dependence_plots-2" src="https://github.com/user-attachments/assets/7293688b-6f07-41c9-845c-506e18342939" />


Key observations from PDPs:
*   **Empty Mass Euro Avg (kg)**: CO2 emissions show a clear increasing trend with vehicle mass.
*   **Maximum Power (kW)**: Similarly, higher power generally corresponds to higher CO2 emissions.
*   **Fuel**: Distinct differences in CO2 levels were observed between fuel types (e.g., diesel vs. petrol).
*   **Gearbox / Body**: Specific gearbox and body configurations also showed varying average CO2 impacts.

### 6. SHAP (SHapley Additive exPlanations) Summary Plot
SHAP values provide a unified measure of feature importance, explaining how each feature contributes to a prediction for individual instances, and aggregated, how they impact the overall model output. The summary plot shows the distribution of SHAP values for each feature across all predictions.

<img width="783" height="933" alt="shap_summary_plot" src="https://github.com/user-attachments/assets/cb1e35a6-e73c-49fa-9c84-39724384d64c" />


The SHAP summary plot confirms that `Empty Mass Euro Avg (kg)` and `Maximum Power (kW)` are the most impactful features, pushing CO2 predictions higher or lower depending on their values. The spread of SHAP values for `Fuel` and `Gearbox` indicates their significant contribution to varying predictions.

### 7. Single Decision Tree Example
To further understand the inner workings of the Random Forest model, an example of a single decision tree (up to a certain depth) from the ensemble was visualized. This shows how the model makes decisions by splitting data based on feature values to arrive at a CO2 prediction.

<img width="1552" height="884" alt="single_tree_plot" src="https://github.com/user-attachments/assets/38507a47-f18d-4bbe-a0eb-fe88778d2be6" />


This visualization demonstrates the hierarchical decision-making process within individual trees, highlighting the sequence of feature splits that lead to a final emission estimate.
