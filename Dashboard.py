import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from fpdf import FPDF

# -------------------------------
# Load dataset fresh each run
# -------------------------------
df = pd.read_csv("C:\\Users\\USER\Desktop\customer segmentation using kmeans clustering\customer_segmentation _dataset.csv")

# -------------------------------
# Add extra features only once
# -------------------------------
if 'Purchase_Frequency' not in df.columns:
    df['Purchase_Frequency'] = np.random.randint(1, 21, size=len(df))

if 'Location' not in df.columns:
    df['Location'] = np.random.choice(['Urban','Suburban','Rural'], size=len(df))

if 'Demographics' not in df.columns:
    df['Demographics'] = np.random.choice(['Student','Professional','Retired','Homemaker'], size=len(df))

# -------------------------------
# Preprocessing
# -------------------------------
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
    df['Gender'] = df['Gender'].map({'male':0, 'female':1, 'm':0, 'f':1})
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0]).astype(int)

# One-Hot Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Location','Demographics'])

# ✅ Ensure no duplicate columns
df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]

# Feature selection
features = ['Annual_Income','Spending_Score','Age','Purchase_Frequency'] + \
           [col for col in df_encoded.columns if 'Location_' in col or 'Demographics_' in col]

scaler = StandardScaler()
df_selected_scaled = scaler.fit_transform(df_encoded[features])

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📌 Dashboard Navigation")
page = st.sidebar.radio("Choose a Page:", 
                        ["🏠 Welcome", 
                         "⚙️ Cluster Selection", 
                         "🔍 Customer Lookup", 
                         "📊 Cluster Visualization", 
                         "📋 Cluster Summary", 
                         "🧮 Confusion Matrix", 
                         "📈 Model Comparison", 
                         "💾 Download Data",
                         "📑 Generate Report"])

# -------------------------------
# Pages
# -------------------------------
if page.startswith("🏠"):
    st.markdown("<h1 style='color:#673AB7;'>✨ Customer Segmentation Dashboard ✨</h1>", unsafe_allow_html=True)

elif page.startswith("⚙️"):
    st.header("⚙️ Cluster Selection")
    k_value = st.number_input("Enter number of clusters (K)", min_value=2, max_value=10, value=4, step=1)
    if st.button("Run Clustering"):
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
        kmeans.fit(df_selected_scaled)
        df_encoded['Cluster'] = kmeans.labels_
        st.session_state['clustered_df'] = df_encoded.copy()
        st.session_state['k_value'] = k_value
        st.success(f"✅ Clustering completed with K = {k_value}. Explore other pages now!")

elif page.startswith("🔍"):
    st.header("🔍 Customer Lookup")
    if 'clustered_df' in st.session_state:
        df_clustered = st.session_state['clustered_df']
        customer_id = st.text_input("Enter Customer ID")
        if customer_id:
            if customer_id in df_clustered['CustomerID'].astype(str).values:
                st.write(df_clustered[df_clustered['CustomerID'].astype(str) == customer_id])
            else:
                st.warning("⚠️ ID not found.")
    else:
        st.warning("⚠️ Run clustering first.")

elif page.startswith("📊"):
    st.header("📊 Cluster Visualization")
    if 'clustered_df' in st.session_state:
        df_clustered = st.session_state['clustered_df']
        fig = px.scatter(df_clustered, x="Annual_Income", y="Spending_Score", color="Cluster",
                         hover_data=['CustomerID','Age','Purchase_Frequency'])
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Run clustering first.")

elif page.startswith("📋"):
    st.header("📋 Cluster Summary")
    if 'clustered_df' in st.session_state:
        df_clustered = st.session_state['clustered_df']
        summary = df_clustered.groupby('Cluster')[features].mean()
        st.dataframe(summary.style.background_gradient(cmap="Blues"))
    else:
        st.warning("⚠️ Run clustering first.")

elif page.startswith("🧮"):
    st.header("🧮 Confusion Matrix")
    if 'clustered_df' in st.session_state:
        df_clustered = st.session_state['clustered_df']
        y_true = df_clustered['Gender']
        y_pred = df_clustered['Cluster']

        accuracy = accuracy_score(y_true, y_pred)
        st.metric("Overall Accuracy", f"{accuracy:.3f}")

        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:", cm)
        st.text(classification_report(y_true, y_pred))

        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax,
                    xticklabels=sorted(set(y_pred)), yticklabels=sorted(set(y_true)))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix Heatmap")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Run clustering first.")

elif page.startswith("📈"):
    st.header("📈 Model Comparison")
    if 'clustered_df' in st.session_state:
        df_clustered = st.session_state['clustered_df']
        y = df_clustered['Gender']
        X = df_clustered[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        }

        results = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({"Model": name, "Accuracy": round(acc,3), "F1 Score": round(f1,3)})

        results_df = pd.DataFrame(results)
        st.session_state['results_df'] = results_df

        st.dataframe(results_df.style.background_gradient(cmap="Greens"))
        fig = px.bar(results_df, x="Model", y=["Accuracy","F1 Score"], barmode="group",
                     title="Model Comparison", color_discrete_sequence=["#2196F3","#FF9800"])
        st.plotly_chart(fig)

elif page.startswith("💾"):
    st.header("💾 Download Data")
    if 'clustered_df' in st.session_state:
        df_clustered = st.session_state['clustered_df']
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Dataset with Cluster Labels", csv, "clustered_data.csv", "text/csv")
    else:
        st.warning("⚠️ Run clustering first.")

elif page.startswith("📑"):
    st.header("📑 Generate Full Report")

    if 'clustered_df' in st.session_state:
        df_clustered = st.session_state['clustered_df']

        # -------------------------------
        # Prepare KPIs
        # -------------------------------
        total_customers = df_clustered['CustomerID'].nunique()
        avg_income = df_clustered['Annual_Income'].mean()
        avg_spending = df_clustered['Spending_Score'].mean()
        overall_accuracy = accuracy_score(df_clustered['Gender'], df_clustered['Cluster'])

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(200, 15, txt="Customer Segmentation Report", ln=True, align="C")
        pdf.ln(15)

        # -------------------------------
        # KPI Summary
        # -------------------------------
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Key Performance Indicators", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt=f"Total Customers: {total_customers}", ln=True)
        pdf.cell(200, 8, txt=f"Average Income: {avg_income:.2f}", ln=True)
        pdf.cell(200, 8, txt=f"Average Spending Score: {avg_spending:.2f}", ln=True)
        pdf.cell(200, 8, txt=f"Overall Accuracy: {overall_accuracy:.2f}", ln=True)
        pdf.ln(15)

        # -------------------------------
        # Cluster Summary
        # -------------------------------
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Cluster Summary", ln=True)
        pdf.set_font("Arial", size=10)
        summary = df_clustered.groupby('Cluster')[features].mean().reset_index()
        for idx, row in summary.iterrows():
            line = f"Cluster {row['Cluster']} | Income: {row['Annual_Income']:.2f} | Spending: {row['Spending_Score']:.2f} | Age: {row['Age']:.2f} | Purchases: {row['Purchase_Frequency']:.2f}"
            pdf.cell(200, 8, txt=line, ln=True)
        pdf.ln(10)

        # -------------------------------
        # Scatter Plot
        # -------------------------------
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Cluster Scatter Plot", ln=True)
        fig1 = px.scatter(df_clustered, x="Annual_Income", y="Spending_Score", color="Cluster",
                          hover_data=['CustomerID','Age','Purchase_Frequency'])
        fig1.write_image("scatter.png")
        pdf.image("scatter.png", x=20, y=None, w=160)
        pdf.ln(15)

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Confusion Matrix Heatmap", ln=True)
        y_true = df_clustered['Gender']
        y_pred = df_clustered['Cluster']
        cm = confusion_matrix(y_true, y_pred)

        fig2, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        plt.savefig("confusion.png")
        plt.close(fig2)
        pdf.image("confusion.png", x=30, y=None, w=120)
        pdf.ln(15)

        # -------------------------------
        # Model Comparison
        # -------------------------------
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Model Comparison Results", ln=True)

        if 'results_df' in st.session_state:
            results_df = st.session_state['results_df']
            pdf.set_font("Arial", size=10)
            for idx, row in results_df.iterrows():
                line = f"{row['Model']} | Accuracy: {row['Accuracy']} | F1: {row['F1 Score']}"
                pdf.cell(200, 8, txt=line, ln=True)
            pdf.ln(10)

            fig3 = px.bar(results_df, x="Model", y=["Accuracy","F1 Score"], barmode="group",
                          title="Model Comparison", color_discrete_sequence=["#2196F3","#FF9800"])
            fig3.write_image("comparison.png")
            pdf.image("comparison.png", x=20, y=None, w=160)
        else:
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt="Model Comparison Results not yet generated.", ln=True)

        # -------------------------------
        # Export PDF
        # -------------------------------
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(
            label="📥 Download Full Dashboard Report",
            data=pdf_bytes,
            file_name="dashboard_report.pdf",
            mime="application/pdf"
        )

    else:
        st.warning("⚠️ Run clustering first.")
