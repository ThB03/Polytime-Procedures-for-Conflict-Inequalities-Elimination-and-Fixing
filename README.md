# Polytime Procedures for Conflict Inequalities, Elimination, and Fixing

This repository contains the code and experimental infrastructure for the paper **"Polytime Procedures for Conflict Inequalities, Elimination, and Fixing"** by Thiago Barbosa and Hamidreza Validi (INFORMS Journal on Computing).

## Overview

We identify a new structure in conflict graphs of MIP formulations with binary decision variables called **hopscotch paths**, which serves as the backbone for polytime procedures to:
- Add conflict inequalities
- Eliminate redundant variables (direct and indirect)
- Fix variables to zero or one

### Key Contribution: Implication Graph Approach

The paper presents two implementations:

| Approach | Implementation | Section | Speed | Use Case |
|----------|---|---------|-------|----------|
| **Conflict Graph + Hopscotch Paths** | `main.py`, `main_gurobi.py` | 4 | Baseline | Theory & reproducibility |
| **Implication Graph + SCCs** | `scc_approach_w_config.py` | 5 | **14–2,820× faster** | **Recommended for production** |

The implication-graph approach achieves one to three orders of magnitude speedups by using strongly connected components (SCCs) and reachability on the SCC condensation DAG.

## Visualization

A conflict graph | Two hopscotch paths
:-------------------------:|:-------------------------:
![](readme_images/conflict_graph_github.PNG?raw=true "A conflict graph") | ![](readme_images/hopscotch_paths_github.PNG?raw=true "Two hopscotch paths")

## Requirements

- **Python 3.11** (required)
- **Gurobi 12.0+** (for Gurobi-based solvers; `main.py` uses free CBC solver)
- See `requirements.txt` for Python dependencies

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# On conda (for NetworkX METIS support, optional):
conda install -c conda-forge networkx-metis