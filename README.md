# agentPathomics

agentPathomics is a modular, agent-orchestrated framework for interpretable morphogenomic analysis in computational pathology. The framework formalizes histomorphometric analysis as a composable, end-to-end methodology that integrates multi-scale image analysis, predictive modeling, and phenotype–genotype association.

## Key Features

- Modular architecture for ROI segmentation, histomorphometric feature extraction, model construction, and morphogenomic analysis  
- Interpretable, histomorphometric representations  
- model development and validation  
- Task-scoped AI agent orchestration operating on structured module inputs and outputs  
- Support for multi-scale histopathological analysis and multi-omic integration  

## Framework Overview

agentPathomics decouples computational pathology workflows into interoperable analytical modules coordinated through a constrained AI-agent layer. The agent interprets task-scoped user instructions (e.g., diagnosis, prognosis, treatment response, exploratory phenotype–genotype mapping) and orchestrates predefined analytical pipelines without performing inference on raw images or molecular data.

## Repository Structure

```text
agentPathomics/
├── segmentation/          # Multi-scale ROI segmentation modules
├── features/              # Histomorphometric feature extractors
├── modeling/              # Feature selection and predictive modeling
├── morphogenomics/        # Path-Genomics Analyzer
├── agent/                 # AI agent orchestration logic
├── configs/               # Configuration files and templates
├── scripts/               # Example workflows and utilities
└── README.md
```

## Installation
For Windows, please install ([Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)) in advance.
```shell
cd agentPathomics
conda env create -f environment.yml
conda env create -f environment_viewer.yml
```
It is recommended to have 64 GB of RAM or more.
Run `stat_all.bat` to start the agent.
