from base64 import b64encode
import sys
sys.path.append("C:\\Users\\kundu\\PycharmProjects\\p\\venv\\Lib\\site-packages")
import streamlit as st
import pandas as pd
import  numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(df,columndrop,fillna_strategy,columns_to_not_transfrom,scalerchoice,encoderchoice):
    # Fill null values with mean for numerical columns
    # Drop selected columns
    df = df.drop(columndrop, axis=1)
    df=df.drop_duplicates()
    df1=df[columns_to_not_transfrom]
    df = df.drop(columns_to_not_transfrom,axis=1)
    numerical_cols = df.select_dtypes(include='number').columns
    ncols = numerical_cols.tolist()
    if ncols:
        if fillna_strategy == "mean":
            imputer = SimpleImputer(strategy='mean')
        elif fillna_strategy == "median":
            imputer = SimpleImputer(strategy='median')
        elif fillna_strategy == "most_frequent":
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            raise ValueError("Invalid fillna strategy selected.")
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

        # Scale numerical columns
        if scalerchoice=='Standard':
            scaler = StandardScaler()
        elif scalerchoice=='MinMax':
            scaler=MinMaxScaler()
        elif scalerchoice=='MaxAbs':
            scaler=MaxAbsScaler()
        elif scalerchoice=='PowerTransformer':
            scaler=PowerTransformer()
        elif scalerchoice=='Quantile':
            rng = np.random.RandomState(0)
            scaler = QuantileTransformer(n_quantiles=10, random_state=0)
        elif scalerchoice=='Robust':
            scaler=RobustScaler()

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    ccols = categorical_cols.tolist()
    if ccols:
        if(encoderchoice=="Label"):
          encoder = LabelEncoder()
          df[categorical_cols] = df[categorical_cols].apply(encoder.fit_transform)
        elif(encoderchoice=="None"):
            encoder = LabelEncoder()
        elif(encoderchoice=="OneHot"):
            encoder=OneHotEncoder()
            enc_data = pd.DataFrame(encoder.fit_transform(df[categorical_cols]).toarray())
            # Merge with main
            df=df.join(enc_data)
    for i in df1.columns:
        extracted_col = df1[i]
        df = df.join(extracted_col)
    return df
def main():

    st.set_page_config(page_title="CSV WIZARD",page_icon=":magic_wand:",layout="wide")

    st.title("CSV File Preprocessor")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        columns_to_not_transfrom = st.multiselect("Select columns not to transfrom", df.columns)
        fillna_strategy = st.selectbox("Select fillna strategy", ["mean", "median", "most_frequent"])
        scalerchoice=st.selectbox("Select the type of scaler",["MinMax","Standard","Robust","PowerTransformer","MaxAbs","Quantile"])
        encoderchoice = st.selectbox("Select the type of scaler",["Label","OneHot","None"])
        # Perform preprocessing
        preprocessed_df = preprocess_data(df,columns_to_drop,fillna_strategy,columns_to_not_transfrom,scalerchoice,encoderchoice)

        # Display preprocessed data
        with st.chat_message("🧑‍💻"):
            st.write("Preprocessed Data")
            st.dataframe(preprocessed_df)

        # Download preprocessed data
        csv = preprocessed_df.to_csv(index=False)
        st.download_button(
            label="Download PreProcessed data",
            data=csv,
            file_name='preprocessed_data.csv',
            mime='text/csv',
        )


if __name__ == "__main__":
    main()
