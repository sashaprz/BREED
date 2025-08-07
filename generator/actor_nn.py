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
# PHASE 1: PRETRAINING COMPONENTS (Run this first to train MPNN on MP data)
# =============================================================================

class MaterialsProjectPretrainer:
    """Pretraining system for MPNN using Materials Project data"""
    
    def __init__(self, api_key: str):
        """
        Initialize with your Materials Project API key
        Get yours at: https://materialsproject.org/api
        """
        self.mpr = MPRester(api_key)
        print(f"Connected to Materials Project with API key: {api_key[:8]}...")
    
    def fetch_electrolyte_materials(self, max_materials: int = 2000):
        """Fetch electrolyte materials from Materials Project"""
        
        criteria = {
            'band_gap': {'$gte': 1.0},  # Insulating/semiconducting
            'formation_energy_per_atom': {'$lte': 0.5},  # Reasonably stable
            'nelements': {'$lte': 5},  # Not too complex
            'elements': {'$in': ['Li', 'Na', 'K', 'F', 'Cl', 'O', 'P', 'S', 'C', 'B', 'Al', 'Si', 'N', 'Ca', 'Mg']}
        }
        
        properties = [
            'material_id', 'formula', 'structure', 'band_gap', 
            'formation_energy_per_atom', 'energy_above_hull', 'density'
        ]
        
        print(f"Fetching up to {max_materials} materials from Materials Project...")
        materials = self.mpr.query(criteria, properties)
        print(f"Found {len(materials)} materials")
        
        return materials[:max_materials]
    
    def structure_to_crystal_graph(self, structure: Structure) -> dgl.DGLGraph:
        """Convert pymatgen Structure to DGL crystal graph"""
        
        # Use CrystalNN to determine coordination environment
        local_env_strategy = local_env.CrystalNN()
        try:
            sg = StructureGraph.with_local_env_strategy(structure, local_env_strategy)
        except:
            # Fallback to simpler method if CrystalNN fails
            from pymatgen.analysis.graphs import StructureGraph
            sg = StructureGraph.with_empty_graph(structure)
        
        node_features = []
        edges = []
        edge_features = []
        
        # Extract node features (atoms)
        for i, site in enumerate(structure.sites):
            atomic_num = site.specie.Z
            
            # Get atomic properties
            try:
                electronegativity = site.specie.X
                ionic_radius = getattr(site.specie, 'ionic_radius', 1.0) or 1.0
                atomic_mass = site.specie.atomic_mass
            except:
                electronegativity = 2.0
                ionic_radius = 1.0
                atomic_mass = 50.0
            
            # Fractional coordinates in unit cell
            frac_coords = site.frac_coords
            
            node_feat = [
                atomic_num,
                electronegativity, 
                ionic_radius,
                atomic_mass / 100.0,  # Normalize
                frac_coords[0],
                frac_coords[1], 
                frac_coords[2]
            ]
            node_features.append(node_feat)
        
        # Extract edges (bonds)
        for i, j, edge_data in sg.graph.edges(data=True):
            if i != j:
                edges.append((i, j))
                distance = edge_data.get('weight', 3.0)
                # Add coordination number and bond type info if available
                coordination = edge_data.get('coordination', 4.0)
                edge_feat = [distance, coordination]
                edge_features.append(edge_feat)
        
        # Create DGL graph
        if not edges:
            # If no bonds detected, add minimal connectivity
            if len(node_features) > 1:
                edges = [(0, 1), (1, 0)]
                edge_features = [[3.0, 4.0], [3.0, 4.0]]
            else:
                edges = []
                edge_features = []
        
        g = dgl.graph(edges if edges else ([], []))
        g.add_nodes(len(node_features))
        
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        g.ndata['atomic_num'] = torch.tensor([int(feat[0]) for feat in node_features], dtype=torch.long)
        
        if edge_features:
            g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
        else:
            g.edata['feat'] = torch.zeros((0, 2), dtype=torch.float32)
        
        return g
    
    def prepare_pretraining_dataset(self, materials_data: List[Dict]):
        """Convert materials to training dataset"""
        
        graphs = []
        targets = []
        
        print("Converting structures to crystal graphs...")
        for i, material in enumerate(materials_data):
            try:
                structure = material['structure']
                graph = self.structure_to_crystal_graph(structure)
                
                target = {
                    'band_gap': material.get('band_gap', 0.0),
                    'formation_energy': material.get('formation_energy_per_atom', 0.0), 
                    'stability': -material.get('energy_above_hull', 0.0),  # More stable = higher value
                    'density': material.get('density', 0.0)
                }
                
                graphs.append(graph)
                targets.append(target)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(materials_data)} materials")
                    
            except Exception as e:
                print(f"Error processing {material.get('formula', 'unknown')}: {e}")
                continue
        
        print(f"Successfully processed {len(graphs)} materials")
        return graphs, targets

class CrystalMPNN(nn.Module):
    """MPNN for crystal structures - this gets pretrained first"""
    
    def __init__(self, 
                 node_feat_dim: int = 7,  # atomic_num, electronegativity, ionic_radius, mass, x, y, z
                 edge_feat_dim: int = 2,  # distance, coordination
                 hidden_dim: int = 128,
                 num_layers: int = 4):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # MPNN layers
        self.mpnn_layers = nn.ModuleList([
            CrystalMPNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Graph pooling
        self.pooling = GlobalAttentionPooling(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        # Property prediction heads (for pretraining)
        self.property_heads = nn.ModuleDict({
            'band_gap': nn.Linear(hidden_dim, 1),
            'formation_energy': nn.Linear(hidden_dim, 1),
            'stability': nn.Linear(hidden_dim, 1),
            'density': nn.Linear(hidden_dim, 1)
        })
    
    def forward(self, g):
        """Forward pass through crystal MPNN"""
        # Encode features
        node_feat = self.node_encoder(g.ndata['feat'])
        edge_feat = self.edge_encoder(g.edata['feat']) if g.num_edges() > 0 else None
        
        # Apply MPNN layers with residual connections
        for layer in self.mpnn_layers:
            node_feat = layer(g, node_feat, edge_feat) + node_feat
        
        # Global pooling to get graph representation
        graph_embedding = self.pooling(g, node_feat)
        
        # Property predictions (for pretraining)
        property_preds = {}
        for prop_name, head in self.property_heads.items():
            property_preds[prop_name] = head(graph_embedding)
        
        return node_feat, graph_embedding, property_preds
    
    def get_graph_embedding(self, g):
        """Get just the graph embedding (for SAC)"""
        _, graph_embedding, _ = self.forward(g)
        return graph_embedding

class CrystalMPNNLayer(nn.Module):
    """Single MPNN layer for crystal structures"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.message_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),  # src + dst + edge
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # node + aggregated messages
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, g, node_feat, edge_feat):
        with g.local_scope():
            g.ndata['h'] = node_feat
            if edge_feat is not None and g.num_edges() > 0:
                g.edata['e'] = edge_feat
                g.apply_edges(self.message_function)
            else:
                g.edata['m'] = torch.zeros((0, node_feat.size(1)))
            
            # Aggregate messages
            g.update_all(dgl.function.copy_e('m', 'm'), dgl.function.sum('m', 'h_agg'))
            
            # Update nodes
            if g.num_nodes() > 0:
                h_agg = g.ndata.get('h_agg', torch.zeros_like(node_feat))
                update_input = torch.cat([node_feat, h_agg], dim=1)
                h_new = self.update_net(update_input)
                return h_new
            else:
                return node_feat
    
    def message_function(self, edges):
        msg_input = torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1)
        return {'m': self.message_net(msg_input)}

def pretrain_crystal_mpnn(api_key: str, save_path: str = "pretrained_crystal_mpnn.pt"):
    """
    PHASE 1: Pretrain the Crystal MPNN on Materials Project data
    Run this first before SAC training!
    """
    
    print("=== PHASE 1: PRETRAINING CRYSTAL MPNN ===")
    
    # Load Materials Project data
    pretrainer = MaterialsProjectPretrainer(api_key)
    materials_data = pretrainer.fetch_electrolyte_materials(max_materials=1000)
    graphs, targets = pretrainer.prepare_pretraining_dataset(materials_data)
    
    if len(graphs) == 0:
        print("No materials loaded! Check your API key and connection.")
        return None
    
    # Initialize MPNN
    mpnn = CrystalMPNN(hidden_dim=128)
    optimizer = torch.optim.Adam(mpnn.parameters(), lr=1e-3)
    
    print(f"Starting pretraining on {len(graphs)} materials...")
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for graph, target in zip(graphs, targets):
            # Forward pass
            _, _, property_preds = mpnn(graph)
            
            # Compute loss
            loss = 0
            loss_count = 0
            for prop_name, pred in property_preds.items():
                if prop_name in target and target[prop_name] is not None:
                    true_val = torch.tensor([[target[prop_name]]], dtype=torch.float32)
                    loss += F.mse_loss(pred, true_val)
                    loss_count += 1
            
            if loss_count > 0:
                loss = loss / loss_count
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        if epoch % 20 == 0 and num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    # Save pretrained model
    torch.save({
        'model_state_dict': mpnn.state_dict(),
        'model_config': {
            'node_feat_dim': 7,
            'edge_feat_dim': 2, 
            'hidden_dim': 128,
            'num_layers': 4
        }
    }, save_path)
    
    print(f"Pretrained MPNN saved to {save_path}")
    return mpnn

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

