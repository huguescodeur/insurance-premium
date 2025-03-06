# # app.py
# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import joblib
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import base64
# from matplotlib.figure import Figure
# import os

# app = Flask(__name__)

# # Charger le modèle
# model = joblib.load('model_insurance.pkl')

# # Charger les données pour la visualisation
# # Supposons que nous avons un fichier CSV avec les données d'assurance
# # Si vous n'avez pas ce fichier, vous pouvez le générer ou le télécharger
# # Vous pouvez également adapter ce code pour utiliser vos propres données
# data = pd.read_csv('insurance.csv') if os.path.exists('insurance.csv') else pd.DataFrame({
#     'age': np.random.randint(18, 65, 1000),
#     'sex': np.random.choice(['male', 'female'], 1000),
#     'bmi': np.random.uniform(18, 40, 1000),
#     'children': np.random.randint(0, 5, 1000),
#     'smoker': np.random.choice(['yes', 'no'], 1000),
#     'region': np.random.choice(['southeast', 'southwest', 'northeast', 'northwest'], 1000),
#     'charges': np.random.uniform(1000, 50000, 1000)
# })

# # Fonction pour créer des graphiques
# def create_plot(plot_type):
#     plt.figure(figsize=(10, 6))
#     plt.tight_layout()
    
#     if plot_type == 'age_vs_premium':
#         plt.scatter(data['age'], data['charges'], alpha=0.5)
#         plt.title('Âge vs Prime d\'assurance')
#         plt.xlabel('Âge')
#         plt.ylabel('Prime d\'assurance (USD)')
#         plt.grid(True, linestyle='--', alpha=0.7)
    
#     elif plot_type == 'bmi_vs_premium':
#         plt.scatter(data['bmi'], data['charges'], alpha=0.5)
#         plt.title('IMC vs Prime d\'assurance')
#         plt.xlabel('Indice de Masse Corporelle (IMC)')
#         plt.ylabel('Prime d\'assurance (USD)')
#         plt.grid(True, linestyle='--', alpha=0.7)
    
#     elif plot_type == 'smoker_distribution':
#         sns.boxplot(x='smoker', y='charges', data=data)
#         plt.title('Distribution des primes par statut de fumeur')
#         plt.xlabel('Fumeur')
#         plt.ylabel('Prime d\'assurance (USD)')
#         plt.grid(True, linestyle='--', alpha=0.7)
    
#     elif plot_type == 'region_distribution':
#         sns.boxplot(x='region', y='charges', data=data)
#         plt.title('Distribution des primes par région')
#         plt.xlabel('Région')
#         plt.ylabel('Prime d\'assurance (USD)')
#         plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Save the plot to a BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     buf.seek(0)
    
#     # Encode the plot to base64 string
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close()
    
#     return img_str

# @app.route('/')
# def home():
#     # Créer les graphiques pour le dashboard
#     age_premium_plot = create_plot('age_vs_premium')
#     bmi_premium_plot = create_plot('bmi_vs_premium')
#     smoker_plot = create_plot('smoker_distribution')
#     region_plot = create_plot('region_distribution')
    
#     return render_template('index.html', 
#                           age_premium_plot=age_premium_plot,
#                           bmi_premium_plot=bmi_premium_plot,
#                           smoker_plot=smoker_plot,
#                           region_plot=region_plot)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Récupérer les données du formulaire
#     gender = request.form.get('gender')
#     smoker = request.form.get('smoker')
#     region = request.form.get('region')
#     age = int(request.form.get('age'))
#     bmi = float(request.form.get('bmi'))
#     children = int(request.form.get('children'))
    
#     # Transformer les entrées pour le modèle
#     gender_val = 1 if gender == 'male' else 0
#     smoker_val = 1 if smoker == 'yes' else 0
    
#     region_val = 0
#     if region == 'southwest':
#         region_val = 1
#     elif region == 'northeast':
#         region_val = 2
#     elif region == 'northwest':
#         region_val = 3
    
#     # Préparer les données pour la prédiction
#     input_data = (age, gender_val, bmi, children, smoker_val, region_val)
#     input_data_array = np.asarray(input_data).reshape(1, -1)
    
#     # Prédire
#     predicted_premium = model.predict(input_data_array)[0]
    
#     # Retourner la prédiction au format JSON
#     return jsonify({
#         'premium': round(predicted_premium, 2),
#         'age': age,
#         'gender': gender,
#         'bmi': bmi,
#         'children': children,
#         'smoker': smoker,
#         'region': region
#     })

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Charger le modèle
model = joblib.load('model_insurance.pkl')

# Charger les données pour la visualisation
data = pd.read_csv('insurance.csv') if os.path.exists('insurance.csv') else pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'sex': np.random.choice(['male', 'female'], 1000),
    'bmi': np.random.uniform(18, 40, 1000),
    'children': np.random.randint(0, 5, 1000),
    'smoker': np.random.choice(['yes', 'no'], 1000),
    'region': np.random.choice(['southeast', 'southwest', 'northeast', 'northwest'], 1000),
    'charges': np.random.uniform(1000, 50000, 1000)
})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    # Préparer les données pour les graphiques interactifs
    
    # Données pour Age vs Premium
    age_data = data.groupby('age')['charges'].mean().reset_index()
    age_premium = {
        'labels': age_data['age'].tolist(),
        'datasets': [{
            'label': 'Prime moyenne ($)',
            'data': age_data['charges'].tolist(),
            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
            'borderColor': 'rgba(54, 162, 235, 1)',
            'borderWidth': 1,
            'pointRadius': 3
        }]
    }
    
    # Données pour BMI vs Premium
    bmi_ranges = [15, 18.5, 25, 30, 35, 40]
    bmi_labels = ['Maigreur', 'Normal', 'Surpoids', 'Obésité I', 'Obésité II']
    
    data['bmi_category'] = pd.cut(data['bmi'], bins=bmi_ranges, labels=bmi_labels)
    bmi_data = data.groupby('bmi_category')['charges'].mean().reset_index()
    
    bmi_premium = {
        'labels': bmi_data['bmi_category'].tolist(),
        'datasets': [{
            'label': 'Prime moyenne ($)',
            'data': bmi_data['charges'].tolist(),
            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            'borderColor': 'rgba(75, 192, 192, 1)',
            'borderWidth': 1
        }]
    }
    
    # Données pour Smoker vs Premium
    smoker_data = data.groupby('smoker')['charges'].mean().reset_index()
    smoker_premium = {
        'labels': smoker_data['smoker'].tolist(),
        'datasets': [{
            'label': 'Prime moyenne ($)',
            'data': smoker_data['charges'].tolist(),
            'backgroundColor': ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)'],
            'borderColor': ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)'],
            'borderWidth': 1
        }]
    }
    
    # Données pour Region vs Premium
    region_data = data.groupby('region')['charges'].mean().reset_index()
    region_premium = {
        'labels': region_data['region'].tolist(),
        'datasets': [{
            'label': 'Prime moyenne ($)',
            'data': region_data['charges'].tolist(),
            'backgroundColor': ['rgba(255, 206, 86, 0.2)', 'rgba(75, 192, 192, 0.2)', 
                               'rgba(153, 102, 255, 0.2)', 'rgba(255, 159, 64, 0.2)'],
            'borderColor': ['rgba(255, 206, 86, 1)', 'rgba(75, 192, 192, 1)', 
                           'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)'],
            'borderWidth': 1
        }]
    }
    
    # Distribution des primes par âge et statut de fumeur
    smoke_age_data = data.groupby(['smoker', pd.cut(data['age'], bins=5)])['charges'].mean().reset_index()
    smoke_age_data.columns = ['smoker', 'age_group', 'charges']
    smoke_age_data['age_group'] = smoke_age_data['age_group'].astype(str)
    
    smoker_yes = smoke_age_data[smoke_age_data['smoker'] == 'yes']['charges'].tolist()
    smoker_no = smoke_age_data[smoke_age_data['smoker'] == 'no']['charges'].tolist()
    age_groups = smoke_age_data[smoke_age_data['smoker'] == 'yes']['age_group'].tolist()
    
    smoke_age_premium = {
        'labels': age_groups,
        'datasets': [
            {
                'label': 'Fumeur',
                'data': smoker_yes,
                'backgroundColor': 'rgba(255, 99, 132, 0.5)',
            },
            {
                'label': 'Non-fumeur',
                'data': smoker_no,
                'backgroundColor': 'rgba(54, 162, 235, 0.5)',
            }
        ]
    }
    
    return jsonify({
        'age_premium': age_premium,
        'bmi_premium': bmi_premium,
        'smoker_premium': smoker_premium,
        'region_premium': region_premium,
        'smoke_age_premium': smoke_age_premium
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    gender = request.form.get('gender')
    smoker = request.form.get('smoker')
    region = request.form.get('region')
    age = int(request.form.get('age'))
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    
    # Transformer les entrées pour le modèle
    gender_val = 1 if gender == 'male' else 0
    smoker_val = 1 if smoker == 'yes' else 0
    
    region_val = 0
    if region == 'southwest':
        region_val = 1
    elif region == 'northeast':
        region_val = 2
    elif region == 'northwest':
        region_val = 3
    
    # Préparer les données pour la prédiction
    input_data = (age, gender_val, bmi, children, smoker_val, region_val)
    input_data_array = np.asarray(input_data).reshape(1, -1)
    
    # Prédire
    predicted_premium = model.predict(input_data_array)[0]
    
    # Comparer avec des profils similaires
    similar_profiles = data[
        (data['age'].between(age-5, age+5)) &
        (data['bmi'].between(bmi-3, bmi+3)) &
        (data['smoker'] == smoker)
    ]
    
    avg_similar = similar_profiles['charges'].mean() if not similar_profiles.empty else predicted_premium
    
    # Retourner la prédiction au format JSON
    return jsonify({
        'premium': round(predicted_premium, 2),
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
        'avg_similar': round(avg_similar, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)