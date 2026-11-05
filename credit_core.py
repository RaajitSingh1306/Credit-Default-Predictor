import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df=pd.read_csv("loan_data(best).csv")

print(df.shape)
print("\n")
print(df.dtypes)
print("\n")
print(df.describe())
print("\n")
print(df.isnull().sum())  #no null
print("\n")
cat_col=['person_gender','person_education',
         'person_home_ownership','loan_intent',
         'previous_loan_defaults_on_file']

for col in cat_col:
    print(f"\n{col}")
    print(df[col].value_counts())
print("\n")
print(df[['person_age','person_income',
        'person_emp_exp','credit_score']].describe())
print("\n")
print(f"Ages above 100: {(df['person_age']>100).sum()} and 80: {(df['person_age']>80).sum()}")
print(f"income above 1M: {(df['person_income']>1_000_000).sum()} and 500k: {(df["person_income"]>500_000).sum()}")
print(f"Emp exp over 60: {(df['person_emp_exp']>60).sum()} and over 40: {(df['person_emp_exp']>40).sum()}")
df['age_exp_gap'] = df['person_age']-df['person_emp_exp']
print("person whose exp=>age:", (df['age_exp_gap']<=0).sum())


original_length=len(df)

#drop age over 100 and exp over 60
df=df[df['person_age']<=100]
df=df[df['person_emp_exp']<=60]
income_cap=df['person_income'].quantile(0.99)
print(f"99th %ile income: {income_cap:,.0f}")
df = df[df['person_income']<=income_cap]

df=df.drop(columns=['age_exp_gap'])
print("\n")
print(f"rows removed:{original_length-len(df)}")
print("rows remaining:", len(df))
print(f"New age range: {(df['person_age']).min()} - {(df['person_age']).max()}")
print(f"New exp range: {(df['person_emp_exp']).min()} - {(df['person_emp_exp']).max()}")
print(f"New income range: {(df['person_income']).min()} - {(df['person_income']).max()}")
print("\n")

cat_cols = ['person_gender', 'person_education', 
            'person_home_ownership', 'loan_intent', 
            'previous_loan_defaults_on_file']

for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\n")

# --- Binary columns: use LabelEncoder (automatic 0/1, no manual map) ---
le = LabelEncoder()
df['person_gender'] = le.fit_transform(df['person_gender'])
print("Gender classes:", le.classes_)  # shows you what got mapped to 0 and 1
df['previous_loan_defaults_on_file'] = le.fit_transform(df['previous_loan_defaults_on_file'])
print("Default classes:", le.classes_)  # verify No=0, Yes=1
# --- Ordinal column: education has a real order ---
edu_order = [['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']]
oe = OrdinalEncoder(categories=edu_order)
df['person_education'] = oe.fit_transform(df[['person_education']])
# --- One-hot: remaining categoricals with no order ---
df = pd.get_dummies(df, columns=['person_home_ownership', 
                                  'loan_intent'], 
                    dtype=int, drop_first=True)

print("\nFinal shape:", df.shape)
print("\n")
print("Columns:", df.columns.tolist())
print("\n")
df = df.drop(columns=['person_emp_exp'])

X = df.drop(columns=['loan_status'])
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")
print(f"\nDefault rate in train:  {y_train.mean():.3f}")
print(f"Default rate in test:   {y_test.mean():.3f}")
print("\n")


# No depth limit — let it grow fully
model_full = DecisionTreeClassifier(random_state=42)
model_full.fit(X_train, y_train)

train_score = model_full.score(X_train, y_train)
test_score  = model_full.score(X_test, y_test)

print(f"Full tree — Train accuracy: {train_score:.3f}")
print(f"Full tree — Test accuracy:  {test_score:.3f}")

# Compare with your shallow tree
model_shallow = DecisionTreeClassifier(max_depth=4, random_state=42)
model_shallow.fit(X_train, y_train)

print(f"\nShallow tree — Train accuracy: {model_shallow.score(X_train, y_train):.3f}")
print(f"Shallow tree — Test accuracy:  {model_shallow.score(X_test, y_test):.3f}")

# What features did the tree find most useful?
feature_importance = pd.Series(
    model_shallow.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(feature_importance.head(10))