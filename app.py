import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Prediksi Gross Tertiary Education Enrollment",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Prediksi Gross Tertiary Education Enrollment")

st.write("""
Aplikasi ini menggunakan algoritma **Random Forest Regression**
untuk memprediksi **Gross Tertiary Education Enrollment**
berdasarkan indikator pendidikan dunia.
""")

# =============================
# MEMBACA DATASET
# =============================
df = pd.read_csv("Global_Education.csv", encoding="latin1")

target = "Gross_Tertiary_Education_Enrollment"

X = df.drop(columns=[target])

X = X.select_dtypes(include=np.number)

y = df[target]

X = X.replace(["..", "...", " "], np.nan)
y = y.replace(["..", "...", " "], np.nan)

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

data = pd.concat([X, y], axis=1)

data = data.dropna(subset=[target])

X = data.drop(columns=[target])
y = data[target]

# =============================
# TRAINING MODEL
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, X.columns)
])

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# =============================
# INPUT USER
# =============================
st.header("Masukkan Nilai Indikator Pendidikan")

user_input = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(X.columns):

    if i % 2 == 0:
        with col1:
            user_input[feature] = st.number_input(
                feature,
                value=float(X[feature].median())
            )

    else:
        with col2:
            user_input[feature] = st.number_input(
                feature,
                value=float(X[feature].median())
            )

# =============================
# PREDIKSI
# =============================
if st.button("Prediksi"):

    input_df = pd.DataFrame([user_input])

    prediction = pipeline.predict(input_df)

    st.success(
        f"Prediksi Gross Tertiary Education Enrollment : {prediction[0]:.2f}%"
    )

# =============================
# FEATURE IMPORTANCE
# =============================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": pipeline.named_steps["model"].feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.subheader("10 Faktor Paling Berpengaruh")

st.dataframe(
    importance.head(10),
    use_container_width=True
)

st.bar_chart(
    importance.head(10).set_index("Feature")
)
