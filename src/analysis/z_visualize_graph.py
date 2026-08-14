import argparse
import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def main():
    # 1. Configuração de Argumentos 
    parser = argparse.ArgumentParser(description="Visualize Biological Network Topology")
    parser.add_argument("--input", required=True, help="Path to the .pkl network file")
    parser.add_argument("--output", required=True, help="Directory to save the image")
    parser.add_argument("--timestamp", required=True, help="Timestamp for the filename")
    args = parser.parse_args()

    print(f"  Desenhando grafo com NetworkX para: {os.path.basename(args.input)}")

    # 2. Carregar os Dados
    if not os.path.exists(args.input):
        print(f"   Erro: Arquivo {args.input} não encontrado.")
        return

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    adj = data.get('adj', None)
    if adj is None:
        print("  Erro: Matriz de adjacência 'adj' não encontrada no .pkl.")
        return

    node_names = data.get('genes', data.get('nodes', [str(i) for i in range(adj.shape[0])]))

    # 3. Criar o Grafo usando NetworkX
    G = nx.DiGraph(adj)
    labels = {i: str(node_names[i]) for i in range(len(node_names))}

    # 4. Configuração Visual do Grafo
    plt.figure(figsize=(14, 14))
    degrees = dict(G.degree())
    node_sizes = [v * 50 + 100 for v in degrees.values()] 

    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', edgecolors='black', alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4, arrowsize=12, edge_color='gray', connectionstyle='arc3, rad=0.1')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', font_color='darkred')

    # 5. Salvar a Imagem
    case_name = os.path.splitext(os.path.basename(args.input))[0]
    plt.title(f"Network Topology: {case_name}\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}", fontsize=16, fontweight='bold')
    plt.axis('off') 
    plt.tight_layout()

    out_path = os.path.join(args.output, f"Graph_{case_name}_{args.timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    main()