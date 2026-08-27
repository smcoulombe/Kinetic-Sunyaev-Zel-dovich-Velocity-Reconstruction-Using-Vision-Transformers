# Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers
Kinetic Sunyaev Zel’dovich (kSZ) velocity reconstruction uses correlations be-
tween maps of the cosmic microwave background (CMB) and a tracer of large
scale structure (LSS) to infer the large-scale peculiar velocity field. Standard
approaches to kSZ velocity reconstruction use quadratic estimators that are Fisher
optimal for purely Gaussian fields. Here, we develop a pipeline for kSZ velocity
reconstruction based on Vision Transformers (ViT) and show that this approach can
utilize non-Gaussian information in the inputs to improve the fidelity of velocity
reconstruction.

---

## Training
* [Training Vision Transformer Model on Gaussian Data]()
* [Finetuning Vision Transformer on Flamingo Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/finetuning_vision_transformer_on_flamingo_data.py)
* [Loss Curves](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/loss_curves.py)

## Results
* [Final Plots](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/final_plots.py)

## Analysis
* [Quadratic Estimator Performance on Gaussian Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/quadratic_estimator_performance_on_gaussian_data.py)
* [Quadratic Estimator Performance on Flamingo Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/quadratic_estimator_performance_on_flamingo_data.py)
* [Vision Transformer without Finetuning Performance on Gaussian Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/training_vision_transformer_model_on_gaussian_data.py)
* [Vision Transformer without Finetuning Performance on Flamingo Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/vision_transformer_without_finetuning_performance_on_flamingo_data.py)
* [Vision Transformer with Finetuning Performance on Flamingo Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/vision_transformer_with_finetuning_performance_on_flamingo_data.py)


## Loss
* [Loss Normalization](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/loss_normalization.py)

## Data
* [Flamingo Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/flamingo_data.py)
* [Generate Gaussian Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/generate_gaussian_data.py)
* [Rotated Flamingo Dataset for Finetuning](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/rotated_flamingo_dataset_for_finetuning.py)
* [Generate CMB Data](https://github.com/smcoulombe/Kinetic-Sunyaev-Zel-dovich-Velocity-Reconstruction-Using-Vision-Transformers/blob/main/generate_cmb_data.py)





