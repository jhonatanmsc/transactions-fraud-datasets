import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def data_cleaning(df: pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = df['amount'].replace('[\$,]', '', regex=True).astype(float)
    df['use_chip'] = df['use_chip'].astype('category')
    df['errors'] = df['errors'].astype('category')


def chart_1(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['amount'], bins=50, kde=True)
    plt.title("Distribuição dos valores das transações")
    plt.xlabel("Valor")
    plt.ylabel("Frequência")
    plt.show()


def chart_2(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='use_chip')
    plt.title("Transações usando chip vs sem chip")
    plt.show()


def chart_3(df):
    top_cities = df['merchant_city'].value_counts().nlargest(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_cities.index, y=top_cities.values)
    plt.xticks(rotation=45)
    plt.title("Top 10 cidades com mais transações")
    plt.ylabel("Número de transações")
    plt.show()


def chart_4(df):
    transactions_per_day = df.groupby(df['date'].dt.date).size()

    plt.figure(figsize=(12, 6))
    transactions_per_day.plot()
    plt.title("Transações por dia")
    plt.xlabel("Data")
    plt.ylabel("Número de transações")
    plt.show()


def chart_5(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='errors', hue='use_chip')
    plt.title("Erros nas transações x Uso do chip")
    plt.xticks(rotation=45)
    plt.show()


def main():
    # Set the path to the file you'd like to load
    file_path = kagglehub.dataset_download("computingvictor/transactions-fraud-datasets") + '/transactions_data.csv'

    # Load the latest version
    df = pd.read_csv(file_path)

    df.info()
    data_cleaning(df)
    print("First 5 records:\n", df.head())

    chart_1(df)
    chart_2(df)
    chart_3(df)
    chart_4(df)
    chart_5(df)


if __name__ == '__main__':
    main()

