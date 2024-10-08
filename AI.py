# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Downloads NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Carregando o dataset (Section 1: Dataset)
df = pd.read_csv('BBC.csv')  # Verifique o formato e o nome correto do arquivo

# Pré-processamento de texto (Section 2: Classification pipeline)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
blacklist = ['said']

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words and word not in blacklist]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# Vetorização TF-IDF para melhorar a qualidade da representação dos textos
vectorizer = TfidfVectorizer(max_features=5000)  # Limitando a 5000 palavras para otimizar performance
X_tfidf = vectorizer.fit_transform(df['clean_text'])

# Dividindo os dados para treino e teste 
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, df['category'], test_size=0.2, stratify=df['category'], random_state=42)

# Definindo os modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB()
}

# Função para treinar e avaliar o modelo com matrizes de confusão e acurácia balanceada
def train_and_evaluate(model, X_train_tfidf, X_test_tfidf, y_train, y_test, num_iterations=10):
    train_accuracies, test_accuracies = [], []
    balanced_accuracies = []
    conf_matrices = np.zeros((len(df['category'].unique()), len(df['category'].unique())))

    for _ in range(num_iterations):
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train_tfidf, y_train, test_size=0.2, stratify=y_train
        )
        
        model.fit(X_train_split, y_train_split)
        
        y_train_pred = model.predict(X_train_split)
        y_test_pred = model.predict(X_test_split)
        
        train_accuracies.append(accuracy_score(y_train_split, y_train_pred))
        test_accuracies.append(accuracy_score(y_test_split, y_test_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test_split, y_test_pred))
        
        conf_matrices += confusion_matrix(y_test_split, y_test_pred)

    mean_train_accuracy = np.mean(train_accuracies)
    mean_test_accuracy = np.mean(test_accuracies)
    mean_balanced_accuracy = np.mean(balanced_accuracies)

    print(f'Mean Training Accuracy: {mean_train_accuracy:.4f}')
    print(f'Mean Test Accuracy: {mean_test_accuracy:.4f}')
    print(f'Mean Balanced Accuracy: {mean_balanced_accuracy:.4f}')
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrices / num_iterations, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=df['category'].unique(), yticklabels=df['category'].unique())
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Função para obter as palavras mais importantes para cada categoria
def get_top_words_per_category(model, vectorizer, X_train_tfidf, y_train, category_label, top_n=5):
    category_index = list(df['category'].unique()).index(category_label)
    
    # Verifica se o modelo é linear e pode gerar coeficientes
    if hasattr(model, 'coef_'):
        category_coef = model.coef_[category_index]
        top_indices = np.argsort(category_coef)[-top_n:]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        print(f"\nTop {top_n} words for category '{category_label}':")
        print(", ".join(top_words))

# Avaliação com os modelos definidos (Section 3: Evaluation)
for name, model in models.items():
    print(f"\n{name} Model:")
    train_and_evaluate(model, X_train_tfidf, X_test_tfidf, y_train, y_test, num_iterations=10)

    # Mostrando as palavras mais importantes para cada categoria
    for category in df['category'].unique():
        get_top_words_per_category(model, vectorizer, X_train_tfidf, y_train, category)

# Análise de Nuvem de Palavras para cada categoria
for category in df['category'].unique():
    text = ' '.join(df[df['category'] == category]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=blacklist).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {category}")
    plt.show()

# Avaliação de desempenho com downsampling do dataset (Section 4: Dataset Size)
sample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
train_errors, test_errors = [], []

for size in sample_sizes:
    X_train_sample, _, y_train_sample, _ = train_test_split(X_train_tfidf, y_train, train_size=size, stratify=y_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_sample, y_train_sample)
    train_errors.append(1 - model.score(X_train_sample, y_train_sample))
    test_errors.append(1 - model.score(X_test_tfidf, y_test))

plt.figure()
plt.plot(sample_sizes, train_errors, label='Train Error')
plt.plot(sample_sizes, test_errors, label='Test Error')
plt.xlabel('Training Set Size Proportion')
plt.ylabel('Error Rate')
plt.title('Error Rate vs Dataset Size')
plt.legend()
plt.show()

# Análise de Tópicos com LDA (Section 5: Topic Analysis)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
X_topics = lda.fit_transform(X_tfidf)

# Visualização dos tópicos
for i, topic in enumerate(lda.components_):
    print(f"Topic {i}:")
    print([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]])

# Classificador de duas camadas (primeiro para tópicos, depois para categorias)
def two_layer_classifier(X_tfidf, X_topics, y_train, y_test):
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_tfidf, y_train, test_size=0.2, stratify=y_train
    )

    # Camada 1: Classificação de tópicos
    lda_classifier = LogisticRegression(max_iter=1000)
    lda_classifier.fit(X_train_split, y_train_split)
    
    # Camada 2: Classificação dentro de cada tópico
    for topic in np.unique(X_topics.argmax(axis=1)):
        idx = X_topics.argmax(axis=1) == topic
        X_topic = X_tfidf[idx]
        y_topic = y_train[idx]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_topic, y_topic)

# Rodando o classificador de duas camadas
two_layer_classifier(X_tfidf, X_topics, y_train, y_test)