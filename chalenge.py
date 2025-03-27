import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sn

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats



from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import root_mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose


#################### Pré processamento de dados #######################################################
df = pd.read_csv("MiningProcess_Flotation_Plant_Database.csv")
df['date'] = pd.to_datetime(df['date'])
# Mudar a marcação de casas decimais para .
# Mudar o tipo de dado para float64

for dt in df.columns[1:]:  
    df[dt] = df[dt].str.replace(',', '.').astype(float)

'''Algumas colunas foram estão amostradas com T = 20s outras com T = 1 h
Para observar Sazonalidade, Tendência e Ruido todo data frame foi reamostrado para T = 20 s'''

'''Filtra os casos onde há menos de 180 registros, que seria o esperado se houvesse uma 
medição a cada 20 segundos em uma hora completa ( 3600 s/ 20s = 180)'''

counts = df.groupby('date').count()
#print(counts[counts['% Iron Feed'] < 180])

hours = pd.Series(df['date'].unique())
hours.index = hours
# crie um índice de data e hora da primeira à última hora incluída na coluna de data
date_range = pd.date_range(start=df.iloc[0,0], end='2017-09-09 23:59:40', freq='20s')

# remove as oberserções que não cumprem com o número mínimo de registro (pode ser visto descomentando o print acima)
date_range = date_range[6:]


hours_list = hours.index.astype(str)
seconds_list = date_range.astype(str)

# Combine o novo índice de data e hora com a série de horas e adicione os timestamps apenas se a data e a hora corresponderem à lista de horas.
new_index = []
for idx in seconds_list:
    if (idx[:13] + ':00:00') in hours_list:
        new_index.append(idx)

# Remove o intervalo faltante dentro da hora encontrada  anteriormente usando as contagens
new_index.remove('2017-04-10 00:00:00')

df['index'] = new_index
df['index'] = pd.to_datetime(df['index'])
df.index = df['index']
df = df.loc[:, df.columns[:-1]]
df.rename(columns={'date': 'datetime hours'}, inplace=True)


############################ EDA ############################
# Verificando quais variáveis têm frequência horária versus frequência de 20 segundos

unique_avg = []
for col in df.columns:
    if col != 'datetime hours':
        unique_avg.append(
            df.groupby('datetime hours')[col].apply(lambda x: len(x.unique())).mean()
        )

plt.plot(np.arange(len(unique_avg)), unique_avg)
plt.title('Average Count of Unique Values per Hour for every Variable')
plt.ylabel('Count')
plt.xticks(list(range(len(unique_avg))), [col for col in df.columns if col != 'datetime hours'], rotation='vertical')
plt.savefig('imagens/frenquencia.png', dpi=200)

# Sondando a hipótese de temporalidade
plt.figure(figsize=(12, 6))

plt.plot(df.index, df['% Iron Feed'], label="% Iron Feed", linestyle='dashed', color='blue')
plt.plot(df.index, df['% Silica Feed'], label="% Silica Feed", linestyle='dashed', color='red')
plt.plot(df.index, df['% Iron Concentrate'], label="% Iron Concentrate", linestyle='solid', color='blue')
plt.plot(df.index, df['% Silica Concentrate'], label="% Silica Concentrate", linestyle='solid', color='red')

plt.xlabel("Tempo")
plt.ylabel("Valores (%)")
plt.title("Séries Temporais das Variáveis do Processo")
plt.legend()
plt.savefig('imagens/temp.png', dpi=200)

# Mapeamento das distribuições dos atributos
a = df.iloc[:, 1:]

plt.figure(figsize=(20,20),dpi=200)
for i , n in enumerate(a.columns.to_list()):
    plt.subplot(6,4,i+1)
    ax = sn.histplot(data=a,x=n, kde=False, bins=20)#, multiple="stack")
    plt.title(f"Histograma {n}", fontdict={"fontsize":14})
    plt.xlabel("")
    plt.ylabel(ax.get_ylabel(), fontdict={"fontsize":12})
    if i not in [0,4,8,12,16,20,24]:
        plt.ylabel("")
    

plt.tight_layout();
plt.savefig('imagens/distribuicao.png', dpi=200)


# Vizualização do Boxplot para mapear outliers
b = df.iloc[:, 1:]
scaler = StandardScaler()
#scaler = MinMaxScaler()
b = pd.DataFrame(scaler.fit_transform(b), columns=b.columns, index=b.index)
b = b.melt()

plt.figure(figsize=(14,6),dpi=200)
sn.boxplot(x=b["variable"], y=b["value"]);
plt.xticks(rotation=90);
plt.xlabel("");
plt.title("Boxplots Univariados");
plt.savefig('imagens/boxplot.png', dpi=200)

# Z-score para mapear outilers
def contar_outliers_zscore(dataframe, limite=3):
    z_scores = np.abs(stats.zscore(dataframe))
    
    outliers_por_coluna = (z_scores > limite).sum()
    
    return outliers_por_coluna

z = df.iloc[:, 1:]
outliers_contagem = contar_outliers_zscore(z)

plt.figure(figsize=(15, 6))
outliers_contagem.plot(kind='bar')
plt.title('Número de Outliers por Feature (Z-Score > 3)')
plt.xlabel('Features')
plt.ylabel('Número de Outliers')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('imagens/z_score.png', dpi=200)

# Dividindo a série em Sazonalidade, Tendencia e Ruído
# Aditiva = Tendencia + Ruido

df = df.drop(columns=[df.columns[0]]) # retirar a coluna "datetime hours"

def decompor_serie(serie, nome_arquivo='imagens/decomposicao.png'):
    resultado = seasonal_decompose(serie, period=180)
    
    plt.figure(figsize=(15,10))
    plt.subplot(411)
    plt.title('Série Original')
    plt.plot(serie)
    
    plt.subplot(412)
    plt.title('Tendência')
    plt.plot(resultado.trend)
    
    plt.subplot(413)
    plt.title('Sazonalidade')
    plt.plot(resultado.seasonal)
    
    plt.subplot(414)
    plt.title('Resíduos (Ruído)')
    plt.plot(resultado.resid)
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=200, bbox_inches='tight')
    
    return resultado

decompor_serie(df['% Silica Concentrate'])

# Checando correlação entre todas as variáveis
plt.figure(figsize=(20, 15))
p = sn.heatmap(df.corr(), annot=True)
plt.savefig('imagens/correlacao_all.png', dpi=200, bbox_inches='tight')

# Divisão do data frame para reduzir a dimensionalidade
# variáveis importantes sugeridas na descrição do data set
imp = df.iloc[:,2:7]
#Entradas
feed = df.iloc[:,0:2]
# Restante das features
air_flow = df.iloc[:,7:14]
level    = df.iloc[:,14:21]
iron_conce = df.iloc[:,21]

plt.figure(figsize=(15, 8))
p = sn.heatmap(feed.corr(), annot=True)
plt.savefig('imagens/correlacao_feed.png', dpi=200, bbox_inches='tight')

plt.figure(figsize=(15, 8))
p = sn.heatmap(imp.corr(), annot=True)
plt.savefig('imagens/correlacao_imp.png', dpi=200, bbox_inches='tight')

plt.figure(figsize=(15, 8))
p = sn.heatmap(air_flow.corr(), annot=True)
plt.savefig('imagens/correlacao_air_flow.png', dpi=200, bbox_inches='tight')

plt.figure(figsize=(15, 8))
p = sn.heatmap(level.corr(), annot=True)
plt.savefig('imagens/correlacao_level.png', dpi=200, bbox_inches='tight')

###### Reduzir, seperadamente, a dimensão do data set por meio do PCA
def apply_pca(data, title,nome_arquivo):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)  # Padronização dos dados

    pca = PCA()
    pca.fit_transform(data_scaled)

    # Variância explicada acumulada
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title(f'Escolha do Número de Componentes PCA ({title})')
    plt.grid()
    plt.savefig(nome_arquivo, dpi=200, bbox_inches='tight')


apply_pca(feed, "Feed",'imagens/feed_pca.png')
apply_pca(air_flow, "Air Flow",'imagens/air_flow_pca.png')
apply_pca(level, "Level",'imagens/level_pca.png')

# Após obter o melhor número de componentes aplica-se o PCA para
# reduzir a demensionaldiade

scaler = StandardScaler()
feed_scaled = scaler.fit_transform(feed)
air_scaled = scaler.fit_transform(air_flow)
level_scaled = scaler.fit_transform(level)
imp_scaled = scaler.fit_transform(imp)
iron_conce_scaled = scaler.fit_transform(iron_conce.values.reshape(-1, 1))


pca_air = PCA(n_components=4)
pca_level = PCA(n_components=4)
pca_feed = PCA(n_components=1)


air_flow_pca = pca_air.fit_transform(air_scaled)
level_pca = pca_level.fit_transform(level_scaled)
feed_pca = pca_feed.fit_transform(feed_scaled)


# Tranformar todos sepadarada em dataframe pandas, pois 
# o PCA retorna um numpy array

pca_air_flow = pd.DataFrame(data = air_flow_pca,columns = ["PCA_air_1","PCA_air_2","PCA_air_3","PCA_air_4"])
pca_level = pd.DataFrame(data = level_pca,columns = ["PCA_level_1","PCA_level_2","PCA_level_3","PCA_level_4"])
pca_feed = pd.DataFrame(data = feed_pca,columns = ["PCA_feed_1"])
imp_scaled = pd.DataFrame(data = imp_scaled,columns = ["Starch Flow","Amina Flow","Ore Pulp Flow","Ore Pulp pH","Ore Pulp Density"])
iron_conce_scaled = pd.DataFrame(data = iron_conce_scaled,columns = ["% Iron Concentrate"])

pca_air_flow = pca_air_flow.reset_index(drop=True)
pca_level = pca_level.reset_index(drop=True)
pca_feed = pca_feed.reset_index(drop=True)
imp_scaled = imp_scaled.reset_index(drop=True)
iron_conce_scaled = iron_conce_scaled.reset_index(drop=True)

pca_combined1 = pd.DataFrame()
pca_combined2 = pd.DataFrame()
pca_combined1 = pd.concat([imp_scaled,pca_air_flow, pca_level, pca_feed,iron_conce_scaled], axis=1)
pca_combined2 = pd.concat([imp_scaled,pca_air_flow, pca_level, pca_feed], axis=1) # Sem % Iron Concentrate


######################### Construção dos Modelos (Validação Cruzada, Treino e Teste) ########################################
X1_pca = pca_combined1
X2_pca = pca_combined2
y = df['% Silica Concentrate']
# Dividir dados em treino e teste
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_pca, y, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_pca, y, test_size=0.2, random_state=42)



def model_trainval(model,x,y,iron_conce):
    
    scoring = ['neg_root_mean_squared_error', 'r2']
    scores = cross_validate(model, x, y, scoring=scoring)
    RMSE = scores['test_neg_root_mean_squared_error'].mean()
    R2 = scores['test_r2'].mean()
    print("Result of model validation" + ' ' + iron_conce + ' ' + "as a feature")
    print(f"RMSE : {RMSE}")
    print(f"R2 :{R2}")
    
    return RMSE,R2

######### Lasso #######################################
reg1=Lasso(alpha=0.001)

print("Métricas do treinamento Lasso")
[RMSE, R2]=model_trainval(reg1,X1_train,y1_train,"with % Iron Concentrate")
[RMSE, R2]=model_trainval(reg1,X2_train,y2_train,"whitout % Iron Concentrate")
print("\n")

###########################################################

################# Ridge ###################################
reg2=Ridge(alpha=0.001)

print("Métricas do treinamento Ridge")
[RMSE, R2]=model_trainval(reg2,X1_train,y1_train,"with % Iron Concentrate")
[RMSE, R2]=model_trainval(reg2,X2_train,y2_train,"whitout % Iron Concentrate")
print("\n")

###########################################################

#################### Random Forest Tree Model ############
reg3=RandomForestRegressor(max_depth=10,n_estimators=10)

print("Métricas do treinamento Random Forest")
[RMSE, R2]=model_trainval(reg3,X1_train,y1_train,"with % Iron Concentrate")
[RMSE, R2]=model_trainval(reg3,X2_train,y2_train,"whitout % Iron Concentrate")
print("\n")

###########################################################

############## MLP ########################################
'''O código abaixo buscar obter melhores hiperâmetros para 
treinamento e teste utilizando MLP. O código está comentado pois 
O random search cross validation é demorado'''


'''mlp = MLPRegressor(max_iter=1000, random_state=42)

param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive']
}

random_search = RandomizedSearchCV(
    mlp, 
    param_distributions=param_dist, 
    cv=3, 
    scoring=['neg_root_mean_squared_error', 'r2'],
    refit='neg_root_mean_squared_error',  # Otimiza baseado no menor RMSE
    n_iter=5, 
    random_state=42,
    verbose=2,
)

random_search.fit(X_train, y_train)

# Recuperar os melhores hiperparâmetros
best_params = random_search.best_params_

# Obter os resultados do modelo com os melhores hiperparâmetros
best_index = random_search.best_index_

mean_rmse = np.mean(random_search.cv_results_['mean_test_neg_root_mean_squared_error'][best_index])
mean_r2 = np.mean(random_search.cv_results_['mean_test_r2'][best_index])

print("\nMelhores hiperparâmetros encontrados:")
print(best_params)
print(f"\nMédia do RMSE dos melhores hiperparâmetros: {mean_rmse:.4f}")
print(f"Média do R² nos melhores hiperparâmetros: {mean_r2:.4f}")'''


# Definição do modelo-
best_params = {
    'solver': 'adam',
    'learning_rate': 'adaptive',
    'hidden_layer_sizes': (200, 100),
    'alpha': 0.001,
    'activation': 'relu'
}

# Treinamento com %Iron Concentrate
mlp1 = MLPRegressor(max_iter=500, random_state=42, verbose=True, **best_params)
mlp1.fit(X1_train, y1_train)
loss1 = mlp1.loss_curve_

y1_train_pred = mlp1.predict(X1_train)
y1_test_pred = mlp1.predict(X1_test)

train_rmse1 = root_mean_squared_error(y1_train, y1_train_pred)
test_rmse1 = root_mean_squared_error(y1_test, y1_test_pred)
train_r21 = r2_score(y1_train, y1_train_pred)
test_r21 = r2_score(y1_test, y1_test_pred)


# Treinamento sem %Iron Concentrate
mlp2 = MLPRegressor(max_iter=500, random_state=42, verbose=True, **best_params)
mlp2.fit(X2_train, y2_train)
loss2 = mlp2.loss_curve_

y2_train_pred = mlp2.predict(X2_train)
y2_test_pred = mlp2.predict(X2_test)

train_rmse2 = root_mean_squared_error(y2_train, y2_train_pred)
test_rmse2 = root_mean_squared_error(y2_test, y2_test_pred)
train_r22 = r2_score(y2_train, y2_train_pred)
test_r22 = r2_score(y2_test, y2_test_pred)


print("\n=== Métricas com %Iron Concentrate ===")
print(f"Train RMSE: {train_rmse1:.4f} | Test RMSE: {test_rmse1:.4f}")
print(f"Train R²  : {train_r21:.4f} | Test R²  : {test_r21:.4f}")

print("\n=== Métricas sem %Iron Concentrate ===")
print(f"Train RMSE: {train_rmse2:.4f} | Test RMSE: {test_rmse2:.4f}")
print(f"Train R²  : {train_r22:.4f} | Test R²  : {test_r22:.4f}")


plt.figure(figsize=(10, 8))
plt.plot(loss1, label='Loss com %Iron Concentrate', color='blue')
plt.plot(loss2, label='Loss sem %Iron Concentrate', color='red')
plt.title('Curva de Perda (Loss) durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('imagens/loss_comparacao.png', dpi=200, bbox_inches='tight')


# Função de plot das séries
def plot_time_series(timesteps, values, start=0, end=None, label=None, color=None, linestyle="-"):
    
    sn.lineplot(x=timesteps[start:end], y=values[start:end], label=label, 
                linewidth=1, alpha=0.7, color=color, linestyle=linestyle)
    
    plt.xlabel("Data")
    plt.ylabel("% Silica Concentrate")
    plt.legend(fontsize=12)
    plt.grid(True)

fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=200, sharex=True)

# Com %Iron Concentrate
plt.sca(axes[0])
plot_time_series(y1_test.index, y1_test_pred, label="Predito", start=100, color="red", linestyle="--")
plot_time_series(y1_test.index, y1_test, label="Real", start=100, color="blue")
plt.title('Predição vs Real (com %Iron Concentrate)')

# Sem %Iron Concentrate
plt.sca(axes[1])
plot_time_series(y2_test.index, y2_test_pred, label="Predito (MLPRregressor)", start=100, color="red", linestyle="--")
plot_time_series(y2_test.index, y2_test, label="Real", start=100, color="blue")
plt.title('Predição vs Real (sem %Iron Concentrate)')

plt.tight_layout()
plt.savefig('imagens/serie_pred_real_duplo.png', dpi=200, bbox_inches='tight')

# Fim do arquivo