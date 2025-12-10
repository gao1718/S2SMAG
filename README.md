# S2SMAG: High-Precision and Generalizable Model for Long-Term Settlement Prediction of Soft Soil Foundations Under Vacuum Preloading

## Citation
If you use the code in this project for academic research, please cite the corresponding paper:
```bibtex
@misc{S2SMAG,
  title={S2SMAG: High-Precision and Generalizable Model for Long-Term Settlement Prediction of Soft Soil Foundations Under Vacuum Preloading},
  author={Gao et al.},
  howpublished={\url{https://github.com/gao1718/S2SMAG}},
  year={2025}
}
````

## Project Introduction
This project open-sources **S2SMAG**, a high-precision intelligent framework tailored for long-term settlement prediction of soft soil foundations under vacuum preloading. It is developed to tackle the core drawbacks of traditional prediction methods—insufficient accuracy and limited generalization capability—and provides critical technical support for the full-life-cycle intelligent construction of soft soil foundation treatment projects.

## Core Features
- **Refined Data Processing**
  Constructs 4D tensor-based training/test sets (dimension: `[target, batch_size, seq_len, input_size]`) via leave-one-out cross-validation, enabling precise separation of target monitoring points (as test set) and remaining monitoring points (as training set).
- **Encoder-Decoder with Cross-Attention Architecture**
  - Encoder integrates **BiGRU** units to realize bidirectional spatiotemporal feature extraction
  - Decoder adopts **GRU** units for long-term sequential settlement prediction
  - Cross-attention mechanism embedded at decoder output to capture the relevance between decoder hidden states and global encoder hidden states, achieving deep spatiotemporal feature mining
- **Enhanced Optimizer Strategy**
  Implements a Lookahead optimization function for stable slow-fast weight updates, which is integrated with the AdamW optimizer during training to improve parameter tuning stability and convergence efficiency.
- **Automated Hyperparameter Optimization**
  Defines hyperparameter search ranges based on Optuna, realizing efficient selection of optimal hyperparameters to maximize model performance.
- **Interpretability Analysis**
  Incorporates SHAP (SHapley Additive exPlanations) tools to conduct interpretability analysis for the deep learning model, revealing key factors dominating settlement prediction results.

## Project Structure
| File Name | Function Description |
|-----------|----------------------|
| `Data_handing.py` | Data processing module; builds 4D tensor training/test sets via leave-one-out cross-validation and partitions target monitoring points (test set) and training samples |
| `seq2seq_attention.py` | Core model implementation; deploys the encoder-decoder architecture (BiGRU encoder + GRU decoder + cross-attention mechanism) |
| `Lookahead_optimizer.py` | Optimizer definition; implements the Lookahead function for integration with AdamW to achieve stable weight updates |
| `Hyperparameter_optimization.py`/`optuna.py` | Hyperparameter optimization; defines search ranges and selects optimal hyperparameters based on Optuna |
| `Model_training.py` | Model training; re-trains the model with optimal hyperparameters and saves training results |
| `Model_testing.py` | Model testing; evaluates the trained model and saves performance metrics |
| `SHAP_DeepExplainer.py` | Model interpretability; performs explainable analysis of the deep learning model using SHAP |


