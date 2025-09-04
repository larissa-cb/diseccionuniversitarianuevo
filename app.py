# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Configuración básica
st.set_page_config(page_title="Predictor Estudiantil", layout="wide")
st.title("🎓 Predictor de Deserción Estudiantil")

# Función simplificada para crear modelo
@st.cache_resource
def create_model():
    np.random.seed(42)
    n_samples = 500
    
    # Datos mínimos para entrenamiento
    data = {
        'Age': np.random.randint(17, 70, n_samples),
        'Admission_grade': np.random.randint(0, 200, n_samples),
        'Courses_1st': np.random.randint(0, 10, n_samples),
        'Approved_1st': np.random.randint(0, 10, n_samples),
        'Grade_1st': np.random.uniform(0, 20, n_samples),
    }
    
    df = pd.DataFrame(data)
    df['Performance'] = df['Approved_1st'] / df['Courses_1st'].replace(0, 1)
    
    # Variable objetivo simple
    df['Target'] = np.where(df['Performance'] < 0.5, 0, 
                           np.where(df['Grade_1st'] < 12, 1, 2))
    
    X = df[['Age', 'Admission_grade', 'Courses_1st', 'Approved_1st', 'Grade_1st', 'Performance']]
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.tolist()

def main():
    model, scaler, features = create_model()
    
    st.sidebar.title("Navegación")
    app_mode = st.sidebar.radio("Modo", ["Predicción", "Info"])
    
    if app_mode == "Predicción":
        st.header("📊 Predicción Individual")
        
        with st.form("form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Edad", 17, 70, 20)
                admission = st.slider("Nota Admisión", 0, 200, 120)
                courses = st.slider("Materias 1er sem", 0, 10, 5)
            
            with col2:
                approved = st.slider("Aprobadas 1er sem", 0, 10, 3)
                grade = st.slider("Promedio 1er sem", 0, 20, 12)
                performance = approved / courses if courses > 0 else 0
            
            if st.form_submit_button("Predecir"):
                input_data = [age, admission, courses, approved, grade, performance]
                input_scaled = scaler.transform([input_data])
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
                
                resultados = ["Abandono", "Enrolado", "Graduado"]
                st.success(f"**Predicción:** {resultados[prediction]}")
                
                fig = go.Figure(go.Bar(x=resultados, y=proba, 
                                     marker_color=['red', 'blue', 'green']))
                st.plotly_chart(fig)
    
    else:
        st.header("ℹ️ Información")
        st.write("App predictora de deserción estudiantil")

if __name__ == "__main__":
    main()