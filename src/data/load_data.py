import pandas as pd 
from pathlib import Path 

def load_dataset(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)

    if not path.exists:
        raise FileNotFoundError(f"File not found in: {path.resolve()}")

    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def check_nulls(df: pd.DataFrame) -> pd.Series:
    number_nulls = df.isnull().sum()
    print("=== Quantidade de Nulos por Coluna ===")
    print(number_nulls[number_nulls > 0])
    print("======================================")

    return number_nulls
