import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_cohort_performance(cohort_metrics, metric_name, title, save_path):
    """
    Plot performance metric across cohorts.
    """
    plt.figure(figsize=(10, 6))
    
    # Reset index to get Cohort name as column
    if isinstance(cohort_metrics, pd.DataFrame):
        df_plot = cohort_metrics.reset_index()
        cohort_col = df_plot.columns[0]
        
        sns.barplot(data=df_plot, x=cohort_col, y=metric_name)
        plt.title(title)
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def generate_visualizations(mcap_perf, vol_perf, results_dir):
    """
    Generate all required visualizations.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Market Cap Plots
    plot_cohort_performance(
        mcap_perf, 
        'execution_rate_low', 
        'Entry (Low) Execution Rate by Market Cap', 
        os.path.join(results_dir, 'mcap_exec_rate_low.png')
    )
    
    plot_cohort_performance(
        mcap_perf, 
        'avg_alpha_low', 
        'Entry (Low) Alpha by Market Cap', 
        os.path.join(results_dir, 'mcap_alpha_low.png')
    )
    
    # Volume Plots
    plot_cohort_performance(
        vol_perf, 
        'execution_rate_low', 
        'Entry (Low) Execution Rate by Volume', 
        os.path.join(results_dir, 'vol_exec_rate_low.png')
    )
