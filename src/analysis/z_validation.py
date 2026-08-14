import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from src.modeling.neural_ode import HybridNeuralODE

# Configuração de estilo para publicação
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def generate_run_id():
    """Gera um identificador único baseado no timestamp atual."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def load_and_evaluate(network_path, model_path, device='cpu'):
    """
    Carrega os dados e o modelo, executa a previsão e calcula métricas.
    """
    if not os.path.exists(network_path) or not os.path.exists(model_path):
        print(f"Arquivos não encontrados: {network_path} ou {model_path}")
        return None

    print(f"   [EVAL] Avaliando: {os.path.basename(model_path)}")

    # 1. Carregar Dados (Grafo + Expressão)
    with open(network_path, 'rb') as f:
        data_pkg = pickle.load(f)
    
    # Preparar Tensores
    adj = torch.FloatTensor(data_pkg['adj'])
    x_data = torch.FloatTensor(data_pkg['x']) # (Time, Genes) ou (Batch, Time, Genes)
    nodes = data_pkg['nodes']
    
    # Ajustar dimensões se necessário (Batch=1)
    if x_data.dim() == 2:
        x_data = x_data.unsqueeze(0) # (1, Time, Genes)
    
    # Split Treino/Teste (simples 80/20) para validação honesta
    n_samples = x_data.shape[0]
    n_train = int(n_samples * 0.8)
    
    # Se tivermos poucas amostras (ex: time-series única), usamos tudo para validar a dinâmica
    if n_samples < 5:
        x_test = x_data
        print("      > Aviso: Dataset pequeno/Time-series única. Validando no dataset completo.")
    else:
        x_test = x_data[n_train:]
        print(f"      > Split Teste: {x_test.shape[0]} amostras.")
        
    x_test = x_test.to(device)
    adj = adj.to(device)
    
    # 2. Inicializar e Carregar Modelo
    n_nodes = len(nodes)
    model = HybridNeuralODE(n_nodes=n_nodes, adj_matrix=adj).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Erro ao carregar pesos do modelo: {e}")
        return None

    # 3. Inferência (Forward Pass)
    # Simula do t=0 ao t=1 (mesmo span do treino)
    n_steps = x_test.shape[1]
    t_span = torch.linspace(0, 1, n_steps).to(device)
    x_init = x_test[:, 0, :] # Estado inicial real
    
    with torch.no_grad():
        x_pred_traj = model(x_init, t_span) # Retorna a trajetória completa
        
    # Pegamos o último ponto para métricas globais (ou a trajetória toda achatada)
    # Flatten para correlação global de todos os pontos
    real_flat = x_test.cpu().numpy().flatten()
    pred_flat = x_pred_traj.cpu().numpy().flatten()
    
    # 4. Cálculo de Métricas Rigorosas
    # Verifica se há variância nos dados para evitar erro de correlação constante
    if np.std(real_flat) > 1e-6 and np.std(pred_flat) > 1e-6:
        pearson_corr, _ = pearsonr(real_flat, pred_flat)
    else:
        pearson_corr = 0.0
        
    mse = mean_squared_error(real_flat, pred_flat)
    
    print(f"      > Pearson R: {pearson_corr:.4f}")
    print(f"      > MSE: {mse:.4f}")
    
    return {
        'real_traj': x_test.cpu().numpy(), # Trajetória completa Real
        'pred_traj': x_pred_traj.cpu().numpy(), # Trajetória completa Predita
        'flat_real': real_flat,
        'flat_pred': pred_flat,
        'nodes': nodes,
        'mse': mse,
        'corr': pearson_corr,
        't_span': t_span.cpu().numpy()
    }

def plot_case_study(results_dict, title, output_filename="validation_plot.png"):
    """
    Gera painel visual completo (Scatter + Dinâmica) para N cenários.
    """
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    n_cases = len(results_dict)
    fig, axes = plt.subplots(2, n_cases, figsize=(6 * n_cases, 10))
    
    # Tratamento para caso único (não array)
    if n_cases == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    if fig.get_suptitle() is None:
        fig.suptitle(title, fontsize=16, y=0.98)

    for i, (scenario_name, res) in enumerate(results_dict.items()):
        if res is None: continue
        
        ax_scatter = axes[0, i]
        ax_traj = axes[1, i]
        
        # --- Plot 1: Scatter Plot (Correlação) ---
        y_true = res['flat_real']
        y_pred = res['flat_pred']
        
        # Downsample se houver muitos pontos para não pesar o PDF
        if len(y_true) > 5000:
            idx = np.random.choice(len(y_true), 5000, replace=False)
            y_true_plot = y_true[idx]
            y_pred_plot = y_pred[idx]
        else:
            y_true_plot, y_pred_plot = y_true, y_pred
            
        ax_scatter.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=5, c='#2c3e50', edgecolor=None)
        
        # Linha de identidade
        min_v = min(y_true_plot.min(), y_pred_plot.min())
        max_v = max(y_true_plot.max(), y_pred_plot.max())
        ax_scatter.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='Ideal')
        
        # Stats Box
        stats_text = (f"Pearson R = {res['corr']:.4f}\n"
                      f"MSE = {res['mse']:.4f}")
        ax_scatter.text(0.05, 0.9, stats_text, transform=ax_scatter.transAxes, 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax_scatter.set_title(f"{scenario_name}\nAjuste Global", fontweight='bold')
        ax_scatter.set_xlabel("Expressão Real")
        ax_scatter.set_ylabel("Expressão Predita (ODE)")

        # --- Plot 2: Dinâmica Temporal (Genes Chave ou Alta Variância) ---
        # Analisa a primeira amostra do teste
        traj_real = res['real_traj'][0] # (Time, Genes)
        traj_pred = res['pred_traj'][0]
        nodes = res['nodes']
        time_axis = res['t_span']
        
        # Seleção Inteligente de Genes:
        # Tenta buscar genes biológicos famosos primeiro (p53, AKT, Insulin), senão usa variância
        famous_genes = ['TP53', 'MDM2', 'CDKN1A', 'AKT1', 'INS', 'INSR', 'CDK1', 'CCNB1', 'BAX']
        selected_indices = [nodes.index(g) for g in famous_genes if g in nodes]
        
        # Se não achou genes famosos suficientes, completa com os de maior variância
        if len(selected_indices) < 3:
            variances = np.var(traj_real, axis=0)
            top_var_indices = np.argsort(variances)[-4:].tolist()
            selected_indices = list(set(selected_indices + top_var_indices))[:4]
            
        colors = sns.color_palette("husl", len(selected_indices))
        
        for j, idx in enumerate(selected_indices):
            gene_name = nodes[idx]
            # Real (Pontos)
            ax_traj.plot(time_axis, traj_real[:, idx], 'o', color=colors[j], alpha=0.3, label=f"{gene_name} (Real)")
            # Predito (Linha Suave)
            ax_traj.plot(time_axis, traj_pred[:, idx], '-', color=colors[j], lw=2, label=f"{gene_name} (ODE)")
            
        ax_traj.set_title("Dinâmica Latente (Amostra #0)")
        ax_traj.set_xlabel("Tempo (Pseudotempo)")
        ax_traj.set_ylabel("Nível de Expressão")
        ax_traj.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"\n Gráfico de Validação Salvo: {save_path}")
    plt.close()