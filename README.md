# Polytime-Procedures-for-Conflict-Inequalities-Elimination-and-Fixing
This is the code and data repository for the paper "Polytime Procedures for Conflict Inequalities, Elimination, and Fixing" by Thiago Barbosa and Hamidreza Validi.

We identify a new structure in the conflict graphs of MIP formulations with binary decision variables, which serves as the backbone of our proposed procedures for finding conflict inequalities, elimination, and fixing. We refer to this new structure as a hopscotch path.

A conflict graph             |  Two hopscotch paths
:-------------------------:|:-------------------------:
![](readme_images/conflict_graph_github.PNG?raw=true "A conflict graph")   |  ![](readme_images/hopscotch_paths_github.PNG?raw=true "Two hopscotch paths")

## Requirements
- Python 3.11 (**required**)
- Required libraries (listed in `requirements.txt`)

## Install Requirements

```
pip install -r requirements.txt
```

## Run

```
python main.py path/to/config.json [--max_vars N]
```

## config.json
The config file can specify a batch of runs. A particular run might look like this:
"entry_name": {
  "filepath": "path/to/problem",
  "category": "<category>",
  "fixing": true,
  "elimination": true
}
