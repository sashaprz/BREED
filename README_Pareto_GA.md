# Multi-Objective Genetic Algorithm for Solid-State Electrolyte Discovery

## Overview

This enhanced genetic algorithm implements **Pareto front optimization** using the **NSGA-II algorithm** to discover optimal solid-state electrolytes for lithium metal batteries. Unlike traditional single-objective optimization, this approach simultaneously optimizes **5 key properties** while maintaining **chemical validity** and **structural feasibility**.

## Key Features

### 🎯 Multi-Objective Optimization
- **5 simultaneous objectives**: Ionic conductivity, bandgap, SEI score, CEI score, bulk modulus
- **Pareto front discovery**: Find trade-offs between competing objectives
- **NSGA-II algorithm**: Non-dominated sorting with crowding distance
- **True multi-objective**: No loss of Pareto-optimal solutions

### 🧪 Enhanced Crystal Generation
- **Charge neutrality**: All generated structures are chemically valid
- **Wyckoff compatibility**: Uses space group multiplicities for realistic structures
- **Diverse element selection**: Broader chemical space exploration
- **Structure validation**: Physical property constraints

### 📊 Comprehensive Analysis
- **Pareto front tracking**: Evolution visualization over generations
- **Hypervolume metrics**: Quantitative convergence assessment
- **Trade-off analysis**: Property correlation insights
- **Rich reporting**: JSON data + visualization plots

## Target Properties

The algorithm optimizes towards these target values for solid-state electrolytes:

| Property | Target Value | Unit | Optimization |
|----------|-------------|------|--------------|
| Ionic Conductivity | 1.0×10⁻³ | S/cm | Minimize error from target |
| Bandgap | 3.0 | eV | Minimize error from target |
| SEI Score | 0.9 | - | Minimize error from target |
| CEI Score | 0.85 | - | Minimize error from target |
| Bulk Modulus | 80 | GPa | Minimize error from target |

## Installation & Requirements

```bash
# Required packages
pip install numpy pandas matplotlib
pip install pymatgen pyxtal
pip install torch  # For ML models

# Your existing ML prediction models should be available
# - main_rl.py (ML property prediction)
# - pyxtal_generation.py (crystal generation utilities)
```

## Usage

### Basic Usage

```python
from genetic_algo_pareto import ParetoFrontGA

# Initialize the genetic algorithm
ga = ParetoFrontGA(
    population_size=80,
    elite_count=6,
    tournament_size=5,
    mutation_rate=0.02,
    max_generations=50,
    convergence_threshold=15,
    output_dir="pareto_electrolyte_ga_results"
)

# Run the optimization
results = ga.run()
```

### Command Line Usage

```bash
python genetic_algo_pareto.py
```

### Advanced Configuration

```python
# Custom target properties
from genetic_algo_pareto import TargetProperties

custom_targets = TargetProperties(
    ionic_conductivity=5.0e-3,  # Higher conductivity target
    bandgap=2.5,                # Lower bandgap
    sei_score=0.95,             # Higher SEI score
    cei_score=0.9,              # Higher CEI score
    bulk_modulus=100.0          # Higher stiffness
)

ga = ParetoFrontGA(
    population_size=100,        # Larger population
    max_generations=100,        # More generations
    convergence_threshold=20,   # Stricter convergence
    output_dir="custom_results"
)

# Override target properties
ga.target_properties = custom_targets
results = ga.run()
```

## Algorithm Details

### NSGA-II Implementation

1. **Non-Dominated Sorting**: Candidates ranked by Pareto dominance
2. **Crowding Distance**: Maintains diversity within each front
3. **Tournament Selection**: Based on rank + crowding distance
4. **Elite Preservation**: Best candidates from each front survive

### Crystal Generation Process

1. **Space Group Selection**: From common electrolyte space groups
2. **Wyckoff Compatibility**: Generate compositions matching multiplicities
3. **Charge Neutrality**: Enforce chemical validity
4. **Structure Generation**: Use PyXtal for realistic structures
5. **Validation**: Check physical constraints

### Multi-Objective Evaluation

Each candidate is evaluated on 5 objectives (all minimization):

```python
objectives = [
    ionic_conductivity_error,  # |log10(actual) - log10(target)|
    bandgap_error,            # |actual - target| / target
    sei_score_error,          # |actual - target|
    cei_score_error,          # |actual - target|
    bulk_modulus_error        # |actual - target| / target
]
```

## Output Structure

```
pareto_electrolyte_ga_results/
├── cifs/                          # Generated CIF files
│   ├── gen0_Li2O4_*.cif
│   └── ...
├── generation_0/
│   ├── population.json            # Full population data
│   └── pareto_fronts.json         # Pareto front analysis
├── generation_1/
│   └── ...
├── pareto_front.png               # Final Pareto front visualization
├── pareto_front_gen10.png         # Intermediate plots
└── final_results.json             # Comprehensive results
```

### Results Data Structure

```json
{
  "generations_run": 25,
  "pareto_front_size": 12,
  "final_hypervolume": 0.847,
  "converged": true,
  "pareto_front_candidates": [
    {
      "composition": {"Li": 4, "O": 4, "P": 2},
      "properties": {
        "ionic_conductivity": 0.00089,
        "bandgap": 3.2,
        "sei_score": 0.87,
        "cei_score": 0.82,
        "bulk_modulus": 78.5
      },
      "objectives": [0.051, 0.067, 0.030, 0.030, 0.019],
      "crowding_distance": 1.23
    }
  ]
}
```

## Visualization

The algorithm generates several types of plots:

1. **3D Pareto Front**: First 3 objectives in 3D space
2. **2D Projections**: Pairwise objective relationships
3. **Hypervolume Evolution**: Convergence tracking
4. **Front Size Evolution**: Diversity metrics

## Performance Considerations

### Computational Requirements
- **Population size**: 80 candidates (recommended)
- **Generations**: 50-100 (depending on convergence)
- **ML evaluations**: ~6,000-16,000 total predictions
- **Runtime**: 2-8 hours (depending on ML model speed)

### Memory Usage
- **CIF files**: ~1-5 MB per generation
- **Population data**: ~10-50 MB total
- **Plots**: ~1-10 MB per generation

### Scaling Recommendations
- **Small runs**: 40 population, 25 generations
- **Standard runs**: 80 population, 50 generations  
- **Large runs**: 120 population, 100 generations

## Convergence Criteria

The algorithm stops when:
1. **Maximum generations** reached, OR
2. **Hypervolume stagnation**: No >1% improvement for 15 generations

### Hypervolume Indicator
- Measures the volume of objective space dominated by the Pareto front
- Higher values indicate better convergence and diversity
- Used for quantitative comparison between runs

## Comparison with Original GA

| Feature | Original GA | Pareto GA |
|---------|-------------|-----------|
| Objectives | Single (weighted sum) | Multiple (5 objectives) |
| Solutions | 1 best candidate | Pareto front (10-20 candidates) |
| Crystal Generation | Hardcoded parameters | Charge-neutral, Wyckoff-aware |
| Selection | Fitness-based | Pareto rank + crowding distance |
| Diversity | Limited | High (crowding distance) |
| Analysis | Basic fitness | Comprehensive trade-off analysis |
| Convergence | Fitness plateau | Hypervolume stability |

## Best Practices

### 1. Parameter Tuning
```python
# For exploration (early stages)
ga = ParetoFrontGA(
    population_size=100,
    mutation_rate=0.05,
    tournament_size=3
)

# For exploitation (refinement)
ga = ParetoFrontGA(
    population_size=80,
    mutation_rate=0.01,
    tournament_size=7
)
```

### 2. Target Property Selection
- Set realistic targets based on literature values
- Consider property correlations and trade-offs
- Adjust targets based on initial population results

### 3. Convergence Monitoring
- Monitor hypervolume evolution
- Check Pareto front diversity
- Analyze property distributions

### 4. Result Analysis
```python
# Load and analyze results
import json
with open('pareto_electrolyte_ga_results/final_results.json', 'r') as f:
    results = json.load(f)

# Find candidates with specific property ranges
good_conductors = [
    c for c in results['pareto_front_candidates']
    if c['properties']['ionic_conductivity'] > 1e-3
]

# Analyze trade-offs
import pandas as pd
df = pd.DataFrame([c['properties'] for c in results['pareto_front_candidates']])
correlation_matrix = df.corr()
```

## Troubleshooting

### Common Issues

1. **Low population diversity**
   - Increase mutation rate
   - Reduce tournament size
   - Check element pool diversity

2. **Slow convergence**
   - Increase population size
   - Adjust target properties
   - Check ML model performance

3. **Invalid structures**
   - Verify PyXtal installation
   - Check space group compatibility
   - Validate charge dictionary

4. **Memory issues**
   - Reduce population size
   - Clean up old CIF files
   - Use smaller max_generations

### Performance Optimization

```python
# For faster testing
ga = ParetoFrontGA(
    population_size=20,
    max_generations=10,
    convergence_threshold=5
)

# For production runs
ga = ParetoFrontGA(
    population_size=80,
    max_generations=50,
    convergence_threshold=15
)
```

## Future Enhancements

1. **Parallel Evaluation**: Multi-process ML predictions
2. **Advanced Operators**: Specialized crossover/mutation
3. **Constraint Handling**: Hard constraints on properties
4. **Interactive Optimization**: User-guided search
5. **Ensemble Methods**: Multiple ML model consensus

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pareto_electrolyte_ga,
  title={Multi-Objective Genetic Algorithm for Solid-State Electrolyte Discovery},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/pareto-electrolyte-ga}
}
```

## License

[Your chosen license]

## Contact

For questions, issues, or contributions, please contact [your email] or open an issue on GitHub.