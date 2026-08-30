[![License https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/LICENSE](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/LICENSE)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)

# Undersampling Research

This repository contains research and analysis on undersampling methods for imbalanced classification datasets.

> **Research status:** The repository is migrating to the leakage-safe v2
> protocol. Existing notebook outputs and conclusions are legacy exploratory
> artifacts and must not be cited as confirmatory findings. See
> [the research-validity protocol](docs/research_validity.md) for the required
> rerun and reporting sequence.

## Overview

We test undersampling methods across various datasets to determine if and when they add value over commonly used ensemble models (i.e., random forests, XGBoost, CatBoost, etc). This research provides empirical evidence to help practitioners decide whether undersampling is beneficial for their specific use cases.

## Results and Discussion

The findings from this research are discussed in detail in our book:

[**Imbalanced Data: Myth, Mistakes and Modern Solutions**](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

[![Book Cover](https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/MOCKUP_BOOK.jpg?raw=true)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

This comprehensive guide explores common misconceptions about handling imbalanced datasets and provides practical, modern solutions based on empirical research.

## Contents

- **Notebooks**: Analysis and evaluation of resampling methods across different datasets and models
- **Functions**: Reusable evaluation and utility functions
- **Models**: Trained models and results from experiments

## Protocol v2 safeguards

- Patient-disjoint train/test splitting for repeated diabetes encounters
- Train-only fitting of imputers, encoders, and feature filters
- Matched successive-halving budgets across baseline and undersampled models
- Training-only OOF decision thresholds, frozen before test evaluation
- Training-only OOF sigmoid calibration, including undersampled models
- Full-test point estimates with 1,000-replicate bootstrap intervals
- Patient-cluster bootstrap where observations are repeated
- Paired model differences and Holm adjustment for comparison families
- Run manifests tied to Git, configuration, environment, and data fingerprints

Create a clean virtual environment, install `requirements.txt`, and run training
only from a clean Git commit. Protocol-v2 scripts intentionally refuse to resume
legacy model folders that do not contain a compatible `run_manifest.json`.
