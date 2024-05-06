import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

pandemic_data = pd.read_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/clean_model_input_and_delphi_parameters.csv')


### Input Distribution
ts_input_columns = ['day_' + str(i) for i in range(30)]

input_distribution = pandemic_data[ts_input_columns + ['population','pandemic']]

x = input_distribution.loc[:,ts_input_columns].div(pandemic_data['population'], axis = 0).values

y = input_distribution.loc[:,['pandemic']].values

x = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2)
PCA_data = pca.fit_transform(X=x)

df = pd.DataFrame(data = PCA_data,
                  columns=['PC1','PC2'])

finalDf = pd.concat([df,input_distribution[['pandemic']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['covid','sars','dengue','ebola','mpox','influenza']
# colors = ['r', 'g', 'b']
for target in targets:
    indicesToKeep = finalDf['pandemic'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               # , c = color
               , s = 50)
ax.legend(targets)
ax.set_ylim(-0.05,0.05)
ax.set_xlim(-1, -0.5)
ax.grid()

fig.savefig('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/input_distribution.png')
print(pca.explained_variance_ratio_)

### Output Distribution
output_columns = ['alpha','days','r_s','r_dth','p_dth','r_dthdecay','k1','k2','jump','t_jump','std_normal','k3']

output_distribution = pandemic_data[output_columns + ['pandemic']]

x = output_distribution.loc[:, output_columns].values
y = output_distribution.loc[:,['pandemic']].values

x = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2)
PCA_data = pca.fit_transform(X=x)

df = pd.DataFrame(data = PCA_data,
                  columns=['PC1','PC2'])

finalDf = pd.concat([df,output_distribution[['pandemic']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['covid','sars','dengue','ebola','mpox','influenza']
# colors = ['r', 'g', 'b']
for target in targets:
    indicesToKeep = finalDf['pandemic'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               # , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

print(pca.explained_variance_ratio_)

fig.savefig('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output_distribution.png')