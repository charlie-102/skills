# Machine Learning Engineer Skills Catalog

100 skills for ML Engineers. Request details on specific skills as needed.

**Also uses:** 25 shared skills from `_shared/catalog.md`

---

## 1. Model Development (15 skills)

| ID | Name | Description |
|----|------|-------------|
| M01 | model-architecture-designer | Design CNN, RNN, Transformer architectures for specific tasks |
| M02 | hyperparameter-tuner | Set up Optuna/Ray Tune for hyperparameter optimization |
| M03 | loss-function-selector | Choose and implement loss functions (Focal, Dice, custom) |
| M04 | optimizer-configurator | Configure AdamW, SGD with proper weight decay settings |
| M05 | layer-debugger | Debug shapes and activations at each layer |
| M06 | gradient-analyzer | Detect vanishing/exploding gradients, plot gradient flow |
| M07 | model-complexity-analyzer | Count params, FLOPs, memory, inference time |
| M08 | attention-visualizer | Visualize attention weights in Transformers |
| M09 | weight-initializer | Apply proper initialization (Kaiming, Xavier, etc.) |
| M10 | regularization-advisor | Recommend dropout, weight decay, label smoothing |
| M11 | batch-size-optimizer | Find max batch size, apply linear scaling rule |
| M12 | learning-rate-scheduler | Configure warmup + cosine annealing schedules |
| M13 | model-pruning | Structured/unstructured pruning for compression |
| M14 | quantization-helper | INT8/FP16 quantization (PTQ and QAT) |
| M15 | knowledge-distillation | Teacher-student training setup |

## 2. Data Pipeline & Processing (12 skills)

| ID | Name | Description |
|----|------|-------------|
| M16 | dataset-loader | Create PyTorch datasets and dataloaders |
| M17 | data-augmentation | Set up augmentation pipelines (albumentations, torchvision) |
| M18 | data-cleaning | Handle corrupt data, duplicates, format issues |
| M19 | feature-engineering | Create features for tabular ML |
| M20 | data-validation | Validate data quality and schema |
| M21 | dataloader-optimizer | Optimize num_workers, prefetch, pin_memory |
| M22 | imbalanced-data-handler | Sampling strategies, class weights for imbalanced data |
| M23 | missing-data-imputer | Handle missing values (imputation strategies) |
| M24 | outlier-detector | Statistical and ML-based outlier detection |
| M25 | data-versioning | Set up DVC for data versioning |
| M26 | synthetic-data-generator | Generate synthetic training data |
| M27 | annotation-helper | Assist with data annotation workflows |

## 3. Training & Experimentation (10 skills)

| ID | Name | Description |
|----|------|-------------|
| M28 | experiment-tracker | Set up MLflow/W&B/TensorBoard tracking |
| M29 | training-monitor | Real-time training metrics and alerts |
| M30 | checkpoint-manager | Save/load checkpoints, resume training |
| M31 | distributed-training | DDP, FSDP, DeepSpeed multi-GPU training |
| M32 | mixed-precision-trainer | AMP training with gradient scaling |
| M33 | early-stopping-config | Configure early stopping with patience |
| M34 | gradient-accumulation | Effective large batches on limited memory |
| M35 | curriculum-learning | Progressive difficulty training schedules |
| M36 | multi-task-trainer | Train on multiple tasks simultaneously |
| M37 | transfer-learning | Fine-tune pretrained models properly |

## 4. MLOps & Deployment (15 skills)

| ID | Name | Description |
|----|------|-------------|
| M38 | model-serving | Deploy with TorchServe, Triton, or FastAPI |
| M39 | onnx-converter | Export models to ONNX format |
| M40 | torchscript-exporter | JIT compile models for production |
| M41 | docker-ml-builder | Create ML-optimized Docker images |
| M42 | kubernetes-ml-deploy | Deploy models on K8s with GPU support |
| M43 | model-registry | Set up MLflow model registry |
| M44 | ab-testing | A/B test model versions in production |
| M45 | canary-deployment | Gradual model rollout strategies |
| M46 | model-versioning | Version control for model artifacts |
| M47 | inference-optimizer | Optimize inference latency (batching, caching) |
| M48 | batch-inference | Set up batch prediction pipelines |
| M49 | streaming-inference | Real-time streaming predictions |
| M50 | model-monitoring | Monitor predictions, latency, errors |
| M51 | drift-detector | Detect data and concept drift |
| M52 | feature-store | Set up Feast or similar feature store |

## 5. Evaluation & Testing (10 skills)

| ID | Name | Description |
|----|------|-------------|
| M53 | metrics-calculator | Compute accuracy, F1, mAP, BLEU, etc. |
| M54 | confusion-matrix-analyzer | Analyze and visualize confusion matrices |
| M55 | roc-curve-plotter | Plot ROC/PR curves with AUC |
| M56 | cross-validation | K-fold and stratified CV setups |
| M57 | statistical-significance | Hypothesis testing for model comparison |
| M58 | model-comparison | Compare multiple models systematically |
| M59 | fairness-evaluator | Evaluate model fairness across groups |
| M60 | robustness-tester | Test model robustness to perturbations |
| M61 | adversarial-tester | Generate adversarial examples |
| M62 | benchmark-runner | Run standard benchmarks (ImageNet, GLUE) |

## 6. GPU & Compute (8 skills)

| ID | Name | Description |
|----|------|-------------|
| M63 | cuda-debugger | Debug CUDA errors and device issues |
| M64 | memory-profiler | Profile GPU memory usage |
| M65 | gpu-utilization-monitor | Monitor GPU util, temp, power |
| M66 | multi-gpu-config | Configure multi-GPU training properly |
| M67 | tpu-optimizer | Optimize code for TPU training |
| M68 | compute-cost-estimator | Estimate cloud compute costs |
| M69 | spot-instance-manager | Use spot/preemptible instances safely |
| M70 | resource-scheduler | Schedule jobs across GPUs/nodes |

## 7. Research & Papers (10 skills)

| ID | Name | Description |
|----|------|-------------|
| M71 | paper-to-code | Implement papers from scratch |
| M72 | arxiv-searcher | Search and summarize arXiv papers |
| M73 | citation-manager | Manage citations and references |
| M74 | reproducibility-checker | Verify reproducibility of results |
| M75 | ablation-study | Design and run ablation studies |
| M76 | baseline-comparator | Set up fair baseline comparisons |
| M77 | sota-tracker | Track state-of-the-art results |
| M78 | paper-summarizer | Summarize ML papers concisely |
| M79 | figure-generator | Create publication-quality figures |
| M80 | latex-helper | Write LaTeX for papers |

## 8. Specific ML Domains (12 skills)

| ID | Name | Description |
|----|------|-------------|
| M81 | cv-pipeline | Computer vision training pipeline |
| M82 | nlp-pipeline | NLP/text processing pipeline |
| M83 | llm-fine-tuner | Fine-tune LLMs (LoRA, QLoRA, full) |
| M84 | rag-builder | Build retrieval-augmented generation systems |
| M85 | embedding-optimizer | Optimize embedding models |
| M86 | prompt-engineer | Design and optimize prompts |
| M87 | diffusion-trainer | Train diffusion models |
| M88 | reinforcement-learning | RL training with stable-baselines/RLlib |
| M89 | time-series-forecaster | Time series forecasting models |
| M90 | recommender-system | Build recommendation systems |
| M91 | anomaly-detector | Anomaly detection models |
| M92 | graph-neural-network | GNN implementation and training |

## 9. Debugging & Optimization (8 skills)

| ID | Name | Description |
|----|------|-------------|
| M93 | nan-debugger | Debug NaN/Inf in training |
| M94 | overfitting-detector | Detect and diagnose overfitting |
| M95 | underfitting-analyzer | Diagnose underfitting issues |
| M96 | convergence-checker | Check if training is converging |
| M97 | bottleneck-finder | Find training/inference bottlenecks |
| M98 | profiler-analyzer | Analyze PyTorch profiler output |
| M99 | memory-leak-detector | Find GPU/CPU memory leaks |
| M100 | numerical-stability | Fix numerical stability issues |

---

## Summary

| Category | Count | IDs |
|----------|-------|-----|
| Model Development | 15 | M01-M15 |
| Data Pipeline | 12 | M16-M27 |
| Training | 10 | M28-M37 |
| MLOps | 15 | M38-M52 |
| Evaluation | 10 | M53-M62 |
| GPU & Compute | 8 | M63-M70 |
| Research | 10 | M71-M80 |
| ML Domains | 12 | M81-M92 |
| Debugging | 8 | M93-M100 |
| **Total** | **100** | |

---

## Request Details

To get detailed implementation for any skill, ask:
> "Give me details on M31 (distributed-training)"

Or multiple:
> "Details on M83, M84, M86 (LLM skills)"
