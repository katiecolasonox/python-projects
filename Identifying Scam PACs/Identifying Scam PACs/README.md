# Description
The package faciliates the execution of multiple K-means clustering and Isolation Forest models. It includes:
- Data Importation: Retrieving and preparing the required datasets (i.e., PAC demographics, PAC expenditures, network graph).
- Exploratory Data Analysis (EDA): Analyzing and visualzing data trends (i.e., distribution of PAC expenditures and outliers).
- Data Cleaning: Processing data to ensure consistency and usabulity (i.e., scaling/normalzing the data to account for large ranges in expenditure categories).
- Modeling: Running K-Means clustering and Isolation Forest algorithms (i.e., ran 100 Monte-Carlo simulations to account for variability in clustering starting points).

This package relies on two custom scripts:
1. isolated_forests.py - Conducts 100 Monte Carlo simulations of Isolated Forests using stratifed sampling to ensure the identifiend "scam PACs" are evenly distriubted through
the test and training subsets. Accruacy and evaulation metrics are also obtained for each Monte Carlo simulation. Averages of these metrics are calculated and recorded to determine which model performed the best. Histograms of the distribuiton of accruacy (true scam PACs identified) are plotted to visualize the precision of the model.
2. k_means_cluster_analysis.py - The same methodology is applied to the K-Means clustering algorithm.
   
# Installation
1. Grouped Expenditures Data - Avaliable in the committees_and_candidates table of the SQL database or downloadable as grouped_expenditures20.csv.
2. Network Graph - Provided in the filed graph_20.nx.pickle
3. FEC Data - Use the fec_load.py script to fetch data direcly from the FEC website. How to load this data is included in the ml_model package. 

# Execution
After gathering the necessary files and scripts, you can execute all models via the included Jupyter Notebook or Python scripts in the ml_model pacakge. Ensure all dependencies
are installed and the environment is correclty set up before running the models.
