from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import ks_2samp

import numpy as np
import pandas as pd
import statistics

import matplotlib.pyplot as plt
import seaborn as sns

class isolation_forest_analysis:
  def __init__(self, all_scaled_data, scam_PAC_names):
    self.all_scaled_data = all_scaled_data
    self.scam_PAC_names = [s.upper() for s in scam_PAC_names]
    self.anomaly_scores_across_simulations = {}
    self.scam_counts = {}

  def run_isolation_forest_monte_carlo(self, all_scaled_data, prob, seed, test_size, features_to_drop, n_simulations):  
    
    """ 
      Runs a monte carlo cluster simulation for isolated forests on a normalized dataset.
      Parameters:
        all_scaled_data: full dataset with scaled features.
        prob: suspected amount of outliers / anomalies (i.e., contamination levels)
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

    # Lists to store results
    train_sil_scores, test_sil_scores = [], []
    train_dbi_scores, test_dbi_scores = [], []
    train_ks_stats, test_ks_stats = [],[]
    train_ks_pvals, test_ks_pvals = [], []
    true_scam_identifications_train, true_scam_identifications_test = [], []

    # Setting up PAC name identification
    PAC_names = all_scaled_data[['CMTE_NM']]
    PAC_names = PAC_names.copy()
    PAC_names.loc[:, 'is_scam_PAC'] = PAC_names['CMTE_NM'].isin(self.scam_PAC_names)

    for i in range(n_simulations):
      current_seed = seed + i

      # Splitting into testing and training data
      train, test = train_test_split(scaled_final, test_size=test_size, random_state=current_seed, stratify=stratify_column)

      # Train model
      iso_forest = IsolationForest(n_estimators=100, contamination=prob, random_state=current_seed)
      iso_forest.fit(train)

      # Anomaly detection for training data
      train = self._process_data(train, iso_forest, PAC_names, 'train')

      # Test model
      iso_forest.fit(test)

      # Anomaly detection for testing data
      test_metrics = self._process_data(test, iso_forest, PAC_names, 'test')

      # Storing the scores for later use
      a_scores = self._store_scores(test)

      # Performance metrics 
      self._collect_metrics(train, test_metrics, train_sil_scores, test_sil_scores, 
                              train_dbi_scores, test_dbi_scores, 
                              train_ks_stats, test_ks_stats, 
                              train_ks_pvals, test_ks_pvals,
                              true_scam_identifications_train, true_scam_identifications_test) 
      
    #Average anomaly scores
    avg_anomaly_scores, std_dev_anomaly_scores = self._calculate_avg_scores(a_scores)

    return self._aggregate_results(train_sil_scores, train_dbi_scores, train_ks_stats, train_ks_pvals, 
                                   true_scam_identifications_train, test_sil_scores, test_dbi_scores, 
                                   test_ks_stats, test_ks_pvals, true_scam_identifications_test) | {'avg_anomaly_scores': avg_anomaly_scores, 'std_dev_anomaly_scores': std_dev_anomaly_scores, 'scam_counts' : self.scam_counts}
  
  def _store_scores(self, data):

    """ 
      Stores the results of the anomaly scores.
      Parameters: 
        data: test or training data from all iterations
    """

    for idx, score in zip(data.index, data['anomaly_score']):
      if idx not in self.anomaly_scores_across_simulations:
          self.anomaly_scores_across_simulations[idx] = []
      self.anomaly_scores_across_simulations[idx].append(score)

    return self.anomaly_scores_across_simulations

  def _calculate_avg_scores(self, scores):
    """ 
     Calculates the average and standard deviations of the anomaly scores. 
      Parameters: 
        scores: anomaly scores from all iterations 
    """
    average_scores = {}
    std_dev_scores = {}

    for idx, scores in self.anomaly_scores_across_simulations.items():
      average_score = sum(scores) / len(scores)
      average_scores[idx] = average_score

      if len(scores) > 1:
        std_dev_score = statistics.stdev(scores)
      else:
        std_dev_score = 0

      std_dev_scores[idx] = std_dev_score
    
    return average_scores, std_dev_scores

  def _process_data(self, data, iso_forest, PAC_names, data_type):
    """ 
     Calculates the accuracy for the true number of scam pacs identified.
      Parameters: 
        data: training and test data,
        iso_forest: fitted isolation forest,
        PAC_names: a list of all the PAC names,
        data_type: a string for 'test' or 'train' data
    """

    # Calculate anomaly scores and thresholds
    data['anomaly_score'] = iso_forest.decision_function(data)
    threshold = data['anomaly_score'].mean() - (3 * data['anomaly_score'].std())

    # Label anomalies
    data['labels'] = np.where(data['anomaly_score'] < threshold, -1, 1)
    data['is_scam'] = np.where(data['labels'] == -1, 1, 0) #1 = scam, 0 = not scam

    # Count occurances of scam labels
    if not hasattr(self, 'scam_counts'):
      self.scam_counts = {}
    for idx, is_scam in zip(data.index, data['is_scam']):
      if idx not in self.scam_counts:
        self.scam_counts[idx] = 0
      self.scam_counts[idx] += is_scam

    # Performance metrics
    sil_score = silhouette_score(data, data['labels'])
    dbi_score = davies_bouldin_score(data, data['labels'])
    
    normal_scores = data['anomaly_score'][data['labels'] == 1]
    anomaly_scores = data['anomaly_score'][data['labels'] == -1]
    ks_stat, ks_pval = ks_2samp(normal_scores, anomaly_scores)

    # True SPAC identification
    names_data = data.join(PAC_names)
    crosstab = pd.crosstab(names_data['is_scam'], names_data['is_scam_PAC'])
    true_scam = crosstab.loc[1, True]
    true_not_scam = crosstab.loc[0, True]
    ratio = true_scam / (true_scam + true_not_scam)

    # Add results to the data object for later use
    if data_type == 'train':
        return sil_score, dbi_score, ks_stat, ks_pval, ratio
    else:
        return sil_score, dbi_score, ks_stat, ks_pval, ratio
    
  def _collect_metrics(self, train, test_metrics, train_sil_scores, test_sil_scores, 
                     train_dbi_scores, test_dbi_scores, 
                     train_ks_stats, test_ks_stats, 
                     train_ks_pvals, test_ks_pvals,
                     true_scam_identifications_train, true_scam_identifications_test):
    
    """ 
      Stores all of the performance metrics.
      Parameters: Metrics from testing and training data
    """

    # Append the metrics from train and test data
    train_sil_scores.append(train[0])
    train_dbi_scores.append(train[1])
    train_ks_stats.append(train[2])
    train_ks_pvals.append(train[3])
    true_scam_identifications_train.append(train[4])
    
    test_sil_scores.append(test_metrics[0])
    test_dbi_scores.append(test_metrics[1])
    test_ks_stats.append(test_metrics[2])
    test_ks_pvals.append(test_metrics[3])
    true_scam_identifications_test.append(test_metrics[4])

  def _aggregate_results(self, train_sil_scores, train_dbi_scores, train_ks_stats, train_ks_pvals, 
                       true_scam_identifications_train, test_sil_scores, test_dbi_scores, 
                       test_ks_stats, test_ks_pvals, true_scam_identifications_test):
    
    """ 
     Aggregates all of the training and test metrics 
      Parameters: 
        data: test or training data
    """

    
    # Return the aggregated results
    aggregated_results =  {
        "average_train_silhouette_score": np.mean(train_sil_scores),
        "std_train_silhouette_score": np.std(train_sil_scores),
        "average_train_davies_bouldin_score": np.mean(train_dbi_scores),
        "std_train_davies_bouldin_score": np.std(train_dbi_scores),
        "average_ks_stat_train": np.mean(train_ks_stats),
        "std_ks_stat_train": np.std(train_ks_stats),
        "average_ks_pvalue_train": np.mean(train_ks_pvals),
        "std_ks_pvalue_train": np.std(train_ks_pvals),
        "average_train_true_scam_identifications": np.mean(true_scam_identifications_train),
        "std_train_true_scam_identifications": np.std(true_scam_identifications_train),

        "average_test_silhouette_score": np.mean(test_sil_scores),
        "std_test_silhouette_score": np.std(test_sil_scores),
        "average_test_davies_bouldin_score": np.mean(test_dbi_scores),
        "std_test_davies_bouldin_score": np.std(test_dbi_scores),
        "average_ks_stat_test": np.mean(test_ks_stats),
        "std_ks_stat_test": np.std(test_ks_stats),
        "average_ks_pvalue_test": np.mean(test_ks_pvals),
        "std_ks_pvalue_test": np.std(test_ks_pvals),
        "average_test_true_scam_identifications": np.mean(true_scam_identifications_test),
        "std_test_true_scam_identifications": np.std(true_scam_identifications_test)
    }

    self.plot_true_scam_distribution(true_scam_identifications_test)

    return aggregated_results
  
  def plot_true_scam_distribution(self, true_scam_identifications_test):
    """ Plot the distribution of the true scam identificantions """

    plt.figure(figsize=(8, 6))
    plt.hist(true_scam_identifications_test, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of True Scam Identifications")
    plt.xlabel("Percentage of True Scam Identification")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
  