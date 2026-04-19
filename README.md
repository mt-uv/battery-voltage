# Battery Voltage Prediction

This repository contains code and data references for predicting battery average voltage using:

- **Graph neural networks (GNNs)** built from charge/discharge crystal structure pairs.
- **RoBERTa fine-tuning** on paired crystal text descriptions.
- **Descriptor-based models** built from magpie material descriptors
- **MLIP relaxation examples** using UMA and MACE.

## Repository structure

- `Graph_ML/`
  - `Graph_Generation.py`: builds paired crystal graphs from CIF files and saves `dual_graph_dataset_.pkl`.
  - `Graph_generation_addition_features.py`: builds paired crystal graphs with additional electronic features and saves `dual_graph_dataset_elect.pkl`.
  - GNN training scripts (`GCN.py`, `GAT.py`, `GATv2.py`, `Transformer_GNN.py`) requires the generated graph .pkl file
- `RoBERTa/`
  - `finetune.py`: fine-tunes dual-branch RoBERTa on charge/discharge crystal descriptions for voltage regression.
- `MLIP/`
  - `eSEN.py`: UMA-based structure relaxation settings.
  - `mace.py`: MACE-based relaxation settings.
- `charge_discharge_pairs.xlsx`
  - Contains charge/discharge material IDs and target `average_voltage`.

## Data requirements

### 1) CIF files from Materials Project (MP)

The graph generation scripts require CIF files from the MP database.

zip file `MP_CIFs_Battery 2.zip` contains all necessary CIF files

### 2) Charge/discharge pair table

`charge_discharge_pairs.xlsx` provides:

- Correct charge/discharge CIF pairs (`id_charge`, `id_discharge`)
- Target voltage (`average_voltage`)

This file is used by all models

### 3) Descriptor-based models

The full descriptor set used for training all descriptor-based model is provided in magpie_descriptors.xlsx

### 4) Crystal descriptions for RoBERTa

`RoBERTa/finetune.py` expects crystal descriptions for charge/discharge pairs.

- The script loads `battery_robocrys_descriptions.csv`.
- Because of file size, these descriptions are distributed separately in figshare

## Workflow

## A) Generate graph datasets (required before GNN training)

1. Prepare `MP_CIFs_Battery/` from MP_CIFs_Battery 2.zip.
2. Ensure `charge_discharge_pairs.xlsx` is present.
3. Run:

```bash
python Graph_ML/Graph_Generation.py
```

Output:

- `dual_graph_dataset_.pkl` (paired graphs + filtered metadata)

This file is required to train the baseline GNN models.


If additional electronic features need to be included in node features,

Run:

```bash
python Graph_ML/Graph_generation_addition_features.py
```

Output:

- `dual_graph_dataset_elect.pkl`

This dataset augments node features with electronic descriptors (e.g., electronegativity, atomic radius, d-electron related features).

## B) Train RoBERTa regression model

1. Donwload `battery_robocrys_descriptions.csv` from figshare.
2. Confirm `charge_discharge_pairs.xlsx` is available.
3. Download checkpoint from figshare to the working environment before running `RoBERTa/finetune.py`.
4. Run:

```bash
python RoBERTa/finetune.py
```

## C) MLIP relaxation examples

- UMA example/settings: `MLIP/eSEN.py`
- MACE example/settings: `MLIP/mace.py`


## Notes

- GPU-enabled environments are recommended.
