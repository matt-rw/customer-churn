import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


dataset_path = '/home/matt/projects/customer-churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'


def load_and_preprocess():
    df = pd.read_csv(dataset_path)

    # Drop customer ID
    df = df.drop(columns=['customerID'])
    # Drop mising
    df = df[df['TotalCharges'] != ' ']
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    # Encode target
    df['Churn'] = LabelEncoder().fit_transform(df['Churn'])  # Yes/No -> 1/0

    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

    # Train-test split
    X = df.drop('Churn', axis=1).values
    y = df['Churn'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val


