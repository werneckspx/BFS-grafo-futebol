import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

def bfs(graph, start_node):
    cor = {u: 'BRANCO' for u in graph.nodes()}
    pi = {u: None for u in graph.nodes()}
    d = {u: np.inf for u in graph.nodes()}

    cor[start_node] = 'CINZA'
    d[start_node] = 0
    pi[start_node] = None

    Q = [start_node]

    while Q:
        u = Q.pop(0)
        for v in graph.neighbors(u):
            if cor[v] == 'BRANCO':
                cor[v] = 'CINZA'
                d[v] = d[u] + 1
                pi[v] = u
                Q.append(v)
        cor[u] = 'PRETO'

    return pi, d

data = pd.read_csv("nome_do_arquivo.csv")

selected_metrics = ['Jogos', 'Gols', 'Assistencias', 'Passes Concluidos', 'Passes Tentados', 'Passes Perigosos', 'BotesDef', 'BotesCentro', 'BotesAta', 'PosseGrandeAreaDef', 'PosseDefesa', 'PosseCentro', 'PosseAtaque', 'PosseGrandeAreaAta']

for metric in selected_metrics:
    if metric != 'Jogos':
        data[metric] = data[metric] / data['Jogos']

data = data[data['Jogos'] > 10]
data = data[data['Posicao'] != 'G']
selected_metrics.remove('Jogos')

data = data[['Jogador'] + selected_metrics]

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[selected_metrics])

similarity_limit = 0  

G = nx.Graph()

jogadores = data['Jogador'].tolist()
G.add_nodes_from(jogadores)

similarity_matrix = np.corrcoef(normalized_data)

for i in range(len(jogadores)):
    for j in range(i + 1, len(jogadores)):
        similarity = similarity_matrix[i][j]
        if similarity > similarity_limit:
            G.add_edge(jogadores[i], jogadores[j])

print("Nós e suas arestas:")
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    print(f"Nó: {node}, Arestas: {neighbors}")

pos = nx.circular_layout(G, 0.8)
plt.figure(figsize=(9, 5))
nx.draw(G, pos, with_labels=True, node_size=300, cmap=plt.cm.Paired, font_weight='bold')
plt.title("Análise de Jogadores de Futebol")
plt.show()
print(G)

starting_player = 'Jogador a ser analisado'

if starting_player in G.nodes():

    pi, d = bfs(G, starting_player)
    print("Vetor de Predecessores:", pi)
    print("Indicativo de distância desde a origem:", d)
    bfs_tree = nx.Graph()
    for node, predecessor in pi.items():
        if predecessor is not None:
            bfs_tree.add_edge(node, predecessor)

    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(bfs_tree) 
    nx.draw(bfs_tree, pos, with_labels=True, node_size=300, font_weight='bold', node_color='skyblue', edge_color='gray')
    plt.title(f'Grafo Resultante da Busca em Largura a partir de {starting_player}')
    plt.show()
else:
    print(f"O jogador {starting_player} não está no grafo.")
    
sns.set(style="white")

print("Matriz")
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=jogadores, yticklabels=jogadores)
plt.title("")
plt.show()