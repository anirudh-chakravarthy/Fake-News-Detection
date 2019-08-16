# Fake-News-Detection-Classifier
The project uses two seperate classifiers to train and evaluate the models for binary and sixway classification tasks on the Liar-Plus dataset (in 2 seperate Jupyter notebooks).

## Methodology
The skeleton architecture of my models are based on the conditions in the dataset's paper (S, SJ, SJ+). The highest validation accuracy was achieved using the parallel Bi-LSTM model on the SJ+ condition, and adding a GRU and Dense Layer. 

## Experiments
The following techniques were experimented/used during training of both classifiers:
1. Hyperparameters: Batch size, Number of units (in LSTM/GRU/Dense), Dropout value, Learning rate.
2. Addition of (multiple) Dropout and Dense layers to the parallel Bi-LSTM.
3. CuDNNLSTM and CuDNNGRU for faster training on GPU.
4. Warm-up training strategy.
5. Bias initialization for minority class handling (sixway classifier).
6. Early Stopping and Reduce LR on plateau.
7. S/SJ/SJ+ conditions implementations as in the paper.

## Results

Classification task | Val accuracy
--- | --- 
Binary classifier | 58.4% 
Sixway Classifier | 24.63%


## References
The following libraries and resources were used in this project:
1. GloVe vector embeddings: https://stackoverflow.com/questions/48677077/how-do-i-create-a-keras-embedding-layer-from-a-pre-trained-word-embedding-datase
2. Multiple inputs for LSTM: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
3. Scikitlearn
4. Tensorflow and keras
5. Numpy 
6. Pandas




