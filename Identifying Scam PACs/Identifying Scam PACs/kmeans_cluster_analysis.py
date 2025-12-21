import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from math import pi

class Kmeans_Cluster_Analysis:
  def __init__(self, scam_PAC_names):
      self.scam_PAC_names = [s.upper() for s in scam_PAC_names]

  def calculate_true_scam_percentage(self, cluster_data, labels, scam_PAC_names, all_scaled_data):
    """ 
      Calculates the true scam percentage scores.
      Parameters: 
        cluster_data: testing or training data
        labels: cluster lables
        scam_PAC_names: a list of potential scam PAC names
        all_scaled_data: all the data (used to get all the PAC names)
    """
    # Label the clusters
    cluster_data['Cluster Label'] = labels
    
    # Join PAC names for scam identification
    PAC_names = all_scaled_data[['CMTE_NM']]
    names = cluster_data.join(PAC_names)
    names['is_scam_PAC'] = names['CMTE_NM'].isin(scam_PAC_names)
    
    # Generate crosstab to identify true scams per cluster
    crosstab = pd.crosstab(names['Cluster Label'], names['is_scam_PAC'])
    
    # Calculate total values for each cluster to identify smallest and largest clusters
    cluster_totals = crosstab.sum(axis=1)
    smallest_cluster = cluster_totals.idxmin()
    largest_cluster = cluster_totals.idxmax()
    
    # Count of true scams in smallest and largest clusters
    true_scams_in_smallest = crosstab.loc[smallest_cluster, True] if True in crosstab.columns else 0
    true_scams_in_largest = crosstab.loc[largest_cluster, True] if True in crosstab.columns else 0
    
    # Calculate the percentage of true scams in the identified clusters
    true_scam_percentage = (true_scams_in_smallest / (true_scams_in_smallest + true_scams_in_largest)) * 100
    
    return true_scam_percentage

  def run_monte_carlo_cluster_model (self, all_scaled_data, test_n, seed, features_to_drop, n_simulations):
      """ 
      Runs a monte carlo cluster simulation for k-means clustering on a normalized dataset.
      Parameters:
        all_scaled_data: full dataset with scaled features.
        test_n: Proportion of the test sample.
        seed: base random seed for reproducibility
        features_to_drop: list of features to exclude from clustering
        n_simulation: number of Monte Carlo simulations to run
      """

      #Features to drop
      scaled_cleaned = all_scaled_data.copy()

      # Stratify column 
      stratify_column = scaled_cleaned['is_known_scam']

      scaled_cleaned.drop(columns = features_to_drop, inplace = True)
      scaled_final = scaled_cleaned.dropna(axis = 0)

      # Lists for storing metrics across simulations
      train_silhouette_scores, train_inertia_scores, train_dbi_scores, train_percentage_of_true_scams_identified_scores = [], [], [], []
      test_silhouette_scores, test_inertia_scores, test_dbi_scores, test_percentage_of_true_scams_identified_scores = [], [], [], []

      for i in range(n_simulations):
         
        current_seed = seed + i 

        # Setting up the data
        train, test = train_test_split(scaled_final, test_size = test_n, random_state = current_seed, stratify=stratify_column) # dividing into train and test

        # Running the clustering model
        kmeans = KMeans(n_clusters = 2, n_init = 10, random_state = current_seed)  # Set the number of clusters
        kmeans.fit(train) # fitting the training data
        train_labels = kmeans.labels_  # getting the cluster labels

        # Training metrics
        train_silhouette = silhouette_score(train, train_labels)
        train_inertia = kmeans.inertia_
        train_dbi = davies_bouldin_score(train, train_labels)

        # Store evulation metrics
        train_silhouette_scores.append(train_silhouette)
        train_inertia_scores.append(train_inertia)
        train_dbi_scores.append(train_dbi)

        # Test the model on the test set
        kmeans.fit(test)
        test_labels = kmeans.labels_

        # Testing metrics
        test_silhouette = silhouette_score(test, test_labels)
        test_inertia = kmeans.inertia_  # test inertia from new k-means fit
        test_dbi = davies_bouldin_score(test, test_labels)

        # Store test metrics
        test_silhouette_scores.append(test_silhouette)
        test_inertia_scores.append(test_inertia)
        test_dbi_scores.append(test_dbi)

        # Calculate the percentage of true scams identified for train and test data
        train_percentage = self.calculate_true_scam_percentage(train, train_labels, self.scam_PAC_names, all_scaled_data)
        test_percentage = self.calculate_true_scam_percentage(test, test_labels, self.scam_PAC_names, all_scaled_data)

        # Store results
        train_percentage_of_true_scams_identified_scores.append(train_percentage)
        test_percentage_of_true_scams_identified_scores.append(test_percentage)
        
      results = {
        "average_train_silhouette_score": round(pd.Series(train_silhouette_scores).mean(), 2),
        "std_train_silhouette_score": round(pd.Series(train_silhouette_scores).std(), 2),
        "average_train_inertia": round(pd.Series(train_inertia_scores).mean(), 2),
        "std_train_inertia": round(pd.Series(train_inertia_scores).std(), 2),
        "average_train_davies_bouldin_index": round(pd.Series(train_dbi_scores).mean(), 2),
        "std_train_davies_bouldin_index": round(pd.Series(train_dbi_scores).std(), 2),
        "average_train_perc_true_scams_identified": round(pd.Series(train_percentage_of_true_scams_identified_scores).mean(), 2),
        "std_train_perc_true_scams_identified" : round(pd.Series(train_percentage_of_true_scams_identified_scores).std(), 2),
        
        "average_test_silhouette_score": round(pd.Series(test_silhouette_scores).mean(), 2),
        "std_test_silhouette_score": round(pd.Series(test_silhouette_scores).std(), 2),
        "average_test_inertia": round(pd.Series(test_inertia_scores).mean(), 2),
        "std_test_inertia": round(pd.Series(test_inertia_scores).std(), 2),
        "average_test_davies_bouldin_index": round(pd.Series(test_dbi_scores).mean(), 2),
        "std_test_davies_bouldin_index": round(pd.Series(test_dbi_scores).std(), 2),
        "average_test_perc_true_scams_identified": round(pd.Series(test_percentage_of_true_scams_identified_scores).mean(), 2),
        "std_test_perc_true_scams_identified" : round(pd.Series(test_percentage_of_true_scams_identified_scores).std(), 2)
      }

      self.plot_true_scam_distribution(test_percentage_of_true_scams_identified_scores)

      return results
  
  def plot_true_scam_distribution(self, test_percentage_of_true_scams_identified_scores):
    """ Plot the distribution of the true scam identificantions """

    plt.figure(figsize=(8, 6))
    plt.hist(test_percentage_of_true_scams_identified_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of True Scam Identifications")
    plt.xlabel("Percentage of True Scam Identification")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
  
  def remove_outliers(self, all_scaled_data, columns):
    """ 
    Given a normalized data set and a list of column names, 
    outliers will be removed from the data set based on the interquartile range.
    """
    for col in columns:
        #Calculate the 25th and 75th quartile
        Q1 = all_scaled_data[col].quantile(0.25)
        Q3 = all_scaled_data[col].quantile(0.75)

        #Calculate the IQR
        IQR = Q3 - Q1

        #Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Filter out the outilers by column 
        no_outlier_data = all_scaled_data[(all_scaled_data[col] >= lower_bound) 
                                                    & (all_scaled_data[col] <= upper_bound)]
    return no_outlier_data




