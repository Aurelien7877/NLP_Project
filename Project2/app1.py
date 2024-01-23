import streamlit as st
import pandas as pd
import torch
import os
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Charger le modèle et le tokenizer
#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")
#model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")

script_directory = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin complet vers le fichier df_QA.json
file_path = os.path.join(script_directory, 'df_QA.json')

df = pd.read_json(file_path, lines=True, convert_axes=False)


st.title("Visualisation d'Embeddings en 3D et Fonctionnalités NLP - Par Maud Tissot et Aurélien Pouxviel")

selected_assureur = st.selectbox("1. Sélectionnez un type d'assureur 🔥 :", df['assureur'].unique())

# Filtrer le DataFrame pour l'assureur sélectionné
df_selected_assureur = df[df['assureur'] == selected_assureur]

st.header(f"1. Visualisation d'Embedding en 3D pour {selected_assureur}")
st.caption("Embeddings fait avec Sbert : paraphrase-MiniLM-L6-v2")
fig = px.scatter_3d(
    df_selected_assureur,
    x=df_selected_assureur['embedding'].apply(lambda x: x[0]),
    y=df_selected_assureur['embedding'].apply(lambda x: x[1]),
    z=df_selected_assureur['embedding'].apply(lambda x: x[2]),
    text=df_selected_assureur.index,
    title=f"Visualisation d'Embedding en 3D pour {selected_assureur}",
    labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'},
)

st.plotly_chart(fig)
####################################
# Ajout de la fonctionnalité de Question Answering
st.header("2. Question Answering ❓🎯​")
st.caption("Modele de question answering d'Hugging face : distilbert-base-cased-distilled-squad")

selected_index_qa = st.selectbox("Sélectionnez une phrase pour le Question Answering :", df_selected_assureur.index, key="qa_selection")

st.write(f"Phrase sélectionnée : {df_selected_assureur.loc[selected_index_qa, 'avis_en']}")

question_qa = st.text_input("Posez une question sur cette phrase (en anglais):", key="qa_question")

if st.button("Répondre à la question"):
    with st.spinner("Chargement du modèle question-answering via hugging face"):
        # Charger le modèle de question-answering
        question_answering_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad')

        # Obtenir la réponse à la question
        result = question_answering_pipeline(
            question=question_qa,
            context=df_selected_assureur.loc[selected_index_qa, 'avis_en']
        )

    st.success("Chargement terminé ✅")

    st.write(f"Réponse : {result['answer']}")
    st.write(f"(Confiance : {result['score']:.4f})")

####################################
# Ajout de la fonctionnalité de Recherche Sémantique
st.header("3. Recherche Sémantique sur l'assureur sélectionné 🔎")
st.caption("Embeddings fait avec Sbert : paraphrase-MiniLM-L6-v2")
st.caption("Puis similarité cosinus entre la query et les avis")

semantic_search_query = st.text_input("Tapper un avis pour trouver des similaires 🔎:", key="search_query")

if st.button("Rechercher"):

    with st.spinner("Chargement du modèle paraphrase-MiniLM-L6-v2..."):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        def generate_embedding(text):
            return model.encode(text)
        
        # Obtenir l'embedding de la recherche sémantique
        query_embedding = generate_embedding(semantic_search_query)

        # Calculer les similarités cosinus entre la recherche et les embeddings des phrases
        similarities_search = util.pytorch_cos_sim(torch.tensor(query_embedding), torch.tensor(df_selected_assureur['embedding'].tolist()))[0]

        df_selected_assureur['similarity_search'] = similarities_search.cpu().numpy()
        sorted_df_search = df_selected_assureur.sort_values(by='similarity_search', ascending=False)


    st.success("Chargement terminé ✅")

    st.write("**Phrases les plus similaires 🏆 :**")
    for index_search, row_search in sorted_df_search.head(3).iterrows():
        st.write(f"- {row_search['avis_en']} (Similarité 📈​: *{row_search['similarity_search']:.4f}*)", unsafe_allow_html=True)