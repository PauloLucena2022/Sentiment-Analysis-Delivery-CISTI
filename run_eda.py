from pathlib import Path 
from src.data.load_data import load_dataset, check_nulls
from src.features.text_stats import add_word_count, get_most_frequent_words
from src.visualization.plots import (plot_word_count_distribution, plot_class_distribution, plot_top_words)

def main():
    raw_data_path = Path("data/raw/delivery_alimentos.csv")
    processed_data_path = Path("data_processed_delivery_alimentos_eda.csv")
    processed_data_path.parent.mkdir(parents = True, exist_ok = True)

    print("1. Loading data...")
    df = load_dataset(raw_data_path)
    check_nulls(df)

    print("2. Calculating textual features...")
    df = add_word_count(df, text_column = 'content')
    plot_word_count_distribution(df)

    print("3. Analysing class distribution (Scores)...")
    plot_class_distribution(df, class_column = 'score')

    print("4. Extracting more frequent words...")
    freq_df = get_most_frequent_words(df, text_column = 'content', top_n = 20)
    print(freq_df.columns)
    plot_top_words(freq_df)

    print("5. Saving Processed Dataset")
    df.to_csv(processed_data_path, index = False)
    print(f"Pipeline completed successfully! Data saved in {processed_data_path}")

if __name__ == "__main__":
    main()