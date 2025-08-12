import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv, GlobalAttentionPooling
import numpy as np
from typing import Dict, List, Tuple, Optional
from pymatgen.core import Structure, Composition
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
import pandas as pd
from collections import defaultdict

# =============================================================================  
# PHASE 2: SAC ACTOR (Uses pretrained MPNN to output crystal graphs)
# =============================================================================

class SACCrystalActor(nn.Module):
    """SAC Actor that takes current crystal graph state and generates new crystal graph"""
    
    def __init__(self, 
                 action_dim: int,
                 pretrained_mpnn_path: str,
                 hidden_dim: int = 128,
                 max_atoms: int = 20):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        
        # Load pretrained Crystal MPNN
        print(f"Loading pretrained MPNN from {pretrained_mpnn_path}")
        checkpoint = torch.load(pretrained_mpnn_path)
        self.crystal_mpnn = CrystalMPNN(**checkpoint['model_config'])
        self.crystal_mpnn.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze MPNN during SAC training (optional - you can unfreeze for fine-tuning)
        for param in self.crystal_mpnn.parameters():
            param.requires_grad = False
        
        # SAC policy network
        # Input: current crystal graph embedding
        # Output: parameters for new crystal graph generation
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        # Crystal graph generator
        self.graph_generator = CrystalGraphGenerator(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_atoms=max_atoms
        )
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, current_crystal_graph):
        """
        SAC Actor forward pass
        
        Args:
            current_crystal_graph: DGL graph of current crystal structure
            
        Returns:
            mean, std: Parameters for action distribution
        """
        # Get embedding of current crystal structure
        current_embedding = self.crystal_mpnn.get_graph_embedding(current_crystal_graph)
        
        # Generate action distribution parameters
        output = self.policy_net(current_embedding)
        mean, log_std = output.chunk(2, dim=-1)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample_action(self, current_crystal_graph):
        """Sample action from policy"""
        mean, std = self.forward(current_crystal_graph)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def generate_new_crystal(self, current_crystal_graph, deterministic=False):
        """
        Generate new crystal structure based on current state
        
        Args:
            current_crystal_graph: Current crystal structure (DGL graph)
            deterministic: Whether to use deterministic policy
            
        Returns:
            new_crystal_graph: DGL graph of new crystal structure
        """
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(current_crystal_graph)
                action = torch.tanh(mean)
            else:
                action, _ = self.sample_action(current_crystal_graph)
            
            # Generate new crystal graph using the action
            new_crystal_graph = self.graph_generator.generate(
                action=action,
                reference_graph=current_crystal_graph
            )
            
            return new_crystal_graph

class CrystalGraphGenerator(nn.Module):
    """Generates new crystal graphs based on continuous actions"""
    
    def __init__(self, action_dim: int, hidden_dim: int, max_atoms: int):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        
        # Common electrolyte elements
        self.electrolyte_elements = [3, 6, 8, 9, 11, 14, 15, 16, 17]  # Li, C, O, F, Na, Si, P, S, Cl
        
        # Action interpretation networks
        self.composition_net = nn.Linear(action_dim, len(self.electrolyte_elements))
        self.structure_net = nn.Linear(action_dim, 4)  # lattice parameters
        self.coordination_net = nn.Linear(action_dim, 1)  # coordination preference
    
    def generate(self, action, reference_graph=None):
        """
        Generate new crystal graph from continuous action
        
        Args:
            action: Continuous action from SAC actor
            reference_graph: Optional reference structure to modify
            
        Returns:
            new_crystal_graph: DGL graph of generated crystal
        """
        action = action.squeeze(0)
        
        # Interpret action for composition
        composition_logits = self.composition_net(action)
        composition_probs = F.softmax(composition_logits, dim=0)
        
        # Interpret action for structure  
        structure_params = torch.sigmoid(self.structure_net(action))
        
        # Generate composition (simplified - sample 2-4 elements)
        num_elements = np.random.randint(2, 5)
        selected_elements = torch.multinomial(composition_probs, num_elements, replacement=False)
        atomic_numbers = [self.electrolyte_elements[i] for i in selected_elements]
        
        # Generate stoichiometry
        stoichiometry = torch.multinomial(torch.ones(len(atomic_numbers)), len(atomic_numbers), replacement=True) + 1
        
        # Create simplified crystal structure
        nodes = []
        node_features = []
        
        for i, (atomic_num, count) in enumerate(zip(atomic_numbers, stoichiometry)):
            for j in range(count.item()):
                nodes.append(atomic_num)
                
                # Simplified features (in practice, use proper crystal structure generation)
                electronegativity = 2.0  # Simplified
                ionic_radius = 1.0
                atomic_mass = atomic_num * 2  # Simplified
                
                # Generate fractional coordinates
                x = (i + j * 0.1) / len(atomic_numbers)
                y = (j * 0.2) % 1.0
                z = (i * 0.3 + j * 0.1) % 1.0
                
                node_feat = [atomic_num, electronegativity, ionic_radius, atomic_mass/100.0, x, y, z]
                node_features.append(node_feat)
        
        # Limit number of atoms
        if len(nodes) > self.max_atoms:
            nodes = nodes[:self.max_atoms]
            node_features = node_features[:self.max_atoms]
        
        # Create bonds based on distance and chemistry
        edges = []
        edge_features = []
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Simplified bonding (in practice, use proper crystal bonding rules)
                coord_i = np.array(node_features[i][4:7])
                coord_j = np.array(node_features[j][4:7])
                distance = np.linalg.norm(coord_i - coord_j)
                
                if distance < 0.5:  # Bond threshold
                    edges.extend([(i, j), (j, i)])
                    edge_features.extend([[distance * 3.0, 4.0], [distance * 3.0, 4.0]])
        
        # Create DGL graph
        g = dgl.graph(edges if edges else ([], []))
        g.add_nodes(len(nodes))
        
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        g.ndata['atomic_num'] = torch.tensor(nodes, dtype=torch.long)
        
        if edge_features:
            g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
        else:
            g.edata['feat'] = torch.zeros((0, 2), dtype=torch.float32)
        
        return g

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def run_complete_pipeline(api_key: str):
    """Complete pipeline: Pretraining → SAC Training"""
    
    # PHASE 1: Pretrain MPNN
    print("Starting Phase 1: Pretraining...")
    pretrained_mpnn = pretrain_crystal_mpnn(api_key, "pretrained_crystal_mpnn.pt")
    
    if pretrained_mpnn is None:
        print("Pretraining failed!")
        return None
    
    # PHASE 2: SAC Training  
    print("\nStarting Phase 2: SAC Training...")
    
    # Initialize SAC actor with pretrained MPNN
    actor = SACCrystalActor(
        action_dim=8,  # Number of generation parameters
        pretrained_mpnn_path="pretrained_crystal_mpnn.pt",
        hidden_dim=128
    )
    
    # Mock SAC training loop
    for episode in range(100):
        # Start with random crystal
        current_crystal = generate_random_crystal()
        
        # Generate new crystal using SAC actor
        new_crystal = actor.generate_new_crystal(current_crystal, deterministic=False)
        
        # Evaluate new crystal (mock reward)
        reward = evaluate_crystal_for_electrolyte(new_crystal)
        
        # SAC updates would happen here
        # (critic updates, actor updates, etc.)
        
        if episode % 20 == 0:
            composition = get_crystal_composition(new_crystal)
            print(f"Episode {episode}: Generated {composition}, Reward: {reward:.3f}")
    
    return actor

def generate_random_crystal():
    """Generate random starting crystal for SAC"""
    # Simplified random crystal generation
    nodes = [3, 9]  # LiF
    node_features = [
        [3, 1.0, 0.76, 6.9/100.0, 0.0, 0.0, 0.0],    # Li
        [9, 4.0, 1.33, 19.0/100.0, 0.5, 0.5, 0.5]    # F
    ]
    edges = [(0, 1), (1, 0)]
    edge_features = [[2.0, 6.0], [2.0, 6.0]]
    
    g = dgl.graph(edges)
    g.add_nodes(2)
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
    g.ndata['atomic_num'] = torch.tensor(nodes, dtype=torch.long)
    g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
    
    return g

def evaluate_crystal_for_electrolyte(crystal_graph):
    """Mock evaluation function"""
    # In practice: DFT calculations, ML predictions, experiments
    composition = get_crystal_composition(crystal_graph)
    reward = 0.0
    
    if 'Li' in composition:
        reward += 2.0
    if 'F' in composition or 'P' in composition:
        reward += 1.5
    
    return reward + np.random.normal(0, 0.1)

def get_crystal_composition(crystal_graph):
    """Extract composition from crystal graph"""
    from collections import Counter
    from pymatgen.core import Element
    
    atomic_nums = crystal_graph.ndata['atomic_num'].tolist()
    composition = Counter(atomic_nums)
    
    comp_str = ""
    for atomic_num, count in composition.items():
        try:
            symbol = Element.from_Z(atomic_num).symbol
            comp_str += f"{symbol}{count if count > 1 else ''}"
        except:
            comp_str += f"Element{atomic_num}_{count}"
    
    return comp_str

if __name__ == "__main__":
    api_key = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    mpnn_model = pretrain_crystal_mpnn(api_key, save_path="pretrained_crystal_mpnn.pt")
    
    print("\nPretraining completed and model saved!")

