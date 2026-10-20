import pandas as pd 
import matplotlib.pyplot as plt
import seaborn 
import seaborn as sns
from pathlib import Path 

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents = True, exist_ok = True)

def plot_word_count_distribution(df: pd.DataFrame):
    plt.figure(figsize = (10,6))
    sns.histplot(df['word_count'], bins = 50)
    plt.title("Distribution of Word Count in Reviews")
    plt.xlabel("Word Count")
    plt.ylabel("Count")

    output_path = FIGURES_DIR / "word_count_histplot.png"
    plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')
    plt.close()
    print(f"Plot saved in: {output_path}")
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def plot_class_distribution(df: pd.DataFrame, class_column: str = 'score'):
    class_counts = df[class_column].value_counts()
    print(f"=== Instance by class ===")
    print(class_counts)
    print("==========================\n")

    # 1. Discret Histogram
    plt.figure(figsize = (8,5))
    sns.histplot(df[class_column], discrete = True)
    plt.title("Histogram of the Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")
    hist_path = FIGURES_DIR / "score_distribution_hist.png"
    plt.savefig(hist_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

    #2. Pie Chart
    plt.figure(figsize = (8,8))
    plt.pie(class_counts, labels = class_counts.index, autopct = "%1.2f%%", startangle = 140)
    plt.title("Distribution of Scores (Pie Chart)")
    plt.axis("equal")
    pie_path = FIGURES_DIR / "score_distribution_pie.png"
    plt.savefig(pie_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

    print(f"Class plots saved in: {FIGURES_DIR}")
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def plot_top_words(freq_df: pd.DataFrame):
    plt.figure(figsize = (12,6))
    sns.barplot(x = 'frequency', y = 'word', data = freq_df, palette = 'viridis')
    plt.title("Top 20 Most Frequent Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")

    output_path = FIGURES_DIR / "top_frequent_words.png"
    plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')
    plt.close()
    print(f"Plot saved in {output_path}")
