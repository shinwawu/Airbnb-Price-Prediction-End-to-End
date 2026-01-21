# 🏠 Airbnb Price Prediction: End-to-End
Este projeto é uma solução completa de Machine Learning para a predição de preços de aluguéis do Airbnb. Ele engloba desde o processamento de dados e treinamento do modelo até a disponibilização de uma API (FastAPI) e uma interface de usuário (Streamlit).

O modelo utiliza um algoritmo de Gradient Boosting Regressor para prever o logaritmo do preço (log_price) com base em características como localização, tipo de propriedade, comodidades e avaliações.

## 📁 Estrutura do Projeto
```
├── app/
│   ├── data/
│   │   └── raw_data.csv       # Dataset original para treino
│   ├── model/
│   │   ├── model.py           # Pipeline de ML e lógica de treino
│   │   └── regressor.pkl      # Modelo serializado (.pkl)
│   └── notebooks/
│       ├── 00-eda.ipynb       # Análise Exploratória de Dados
│       └── 01-data_model.ipynb # Prototipagem do modelo
├── main.py                    # API principal com FastAPI
├── streamlit_app.py           # Interface Frontend Streamlit
├── requirements.txt           # Dependências do projeto
├── Dockerfile                 # Configuração da imagem Docker
├── docker-compose.yml         # Orquestração dos containers
└── README.md                  # Documentação
```
## 🚀 Tecnologias Utilizadas
Python 3.9+
Scikit-Learn: Pipelines, pré-processamento e Gradient Boosting.
Pandas & Numpy: Manipulação de dados.
FastAPI: Criação da API de predição.
Streamlit: Interface frontend intuitiva.
Joblib: Persistência do modelo treinado.
Docker & Docker Compose: Para containerização e deploy simplificado.

## 🛠️ Como Executar o Projeto. 
Clonar o repositório e Instalar Dependências
```
bash
git clone https://github.com/seu-usuario/airbnb-price-prediction.git
cd airbnb-price-prediction
pip install -r requirements.txt
```

Como Executar com Docker
Certifique-se de ter o Docker instalado.
```
bash
docker-compose up --build
```

Acesse:
Interface Web (Streamlit): http://localhost:8501

Como executar sem Docker

1.Executar a API (Backend)
A API é responsável por carregar o modelo e processar as requisições de predição. Se o modelo (regressor.pkl) não existir, o sistema executará o treino automaticamente no primeiro acesso.
```
bash
uvicorn main:app --reload
```

2. Executar o Streamlit (Frontend)
Em um novo terminal, execute a interface visual:
```
bash
streamlit run streamlit_app.py
```

## 🧠 Detalhes do Modelo
O pipeline de dados foi desenhado para lidar com diferentes tipos de variáveis:

Numéricas: Preenchimento de valores ausentes pela mediana e padronização (StandardScaler). Inclui accommodates, bathrooms, latitude, longitude, entre outras.

Categóricas: Preenchimento com o valor mais frequente e codificação via OneHotEncoder.

Amenities: Uma função customizada de normalização limpa e ordena as strings de comodidades para garantir consistência entre o treino e a predição.

## 📬 Endpoints da API
Método,Endpoint,Descrição
GET,/health,Verifica se a API está online.
POST,/predict,Recebe os dados do imóvel e retorna a predição do log_price.

Exemplo de Payload (JSON):
{
  "accommodates": [4.0],
  "bathrooms": [1.0],
  "property_type": ["Apartment"],
  "amenities": ["{Wifi,Kitchen}"]
}

## 📝 Notas Adicionais
Target: O modelo prevê o valor em logaritmo. No streamlit_app.py, você pode descomentar a linha math.exp(y_pred) para visualizar o valor real em moeda.

Treinamento: O treinamento é engatilhado via API caso o arquivo .pkl não seja encontrado, utilizando o raw_data.csv presente na pasta data.
