import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Buat kolom status kelulusan: nilai rata-rata >= 60 dianggap "Lulus"
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['status_lulus'] = df['average_score'].apply(lambda x: 1 if x >= 60 else 0)

    # Encoding kolom kategorikal
    df_encoded = pd.get_dummies(df[['gender', 'race/ethnicity', 'parental level of education',
                                     'lunch', 'test preparation course']], drop_first=True)

    # Gabungkan dengan fitur numerik
    X = pd.concat([df_encoded, df[['math score', 'reading score', 'writing score']]], axis=1)
    y = df['status_lulus']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42, max_depth=4)
    clf.fit(X_train, y_train)
    return clf
