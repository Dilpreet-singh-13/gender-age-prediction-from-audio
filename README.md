# gender-age-prediction-from-audio
A Deep learning model that predicts the gender and age using audio clips

## Dataset
The dataset used in this project is derived from Mozilla Common Voice, which provides a large collection of speech data contributed by volunteers. The dataset has been preprocessed to ensure balanced representation of male and female genders for improved model performance. [Mozilla Common Voice](https://commonvoice.mozilla.org/en/)

**Columns used**
- `filename`: Path to the audio file
- `gender`: Encoded as 0 for male, 1 for female (others are discarded)
- `age`: Mapped categorical values like “twenties” → 20, “thirties” → 30, etc.


## Data Cleaning and Balancing
Dropped Columns: `up_votes`, `down_votes`, `accent`, `duration`, `text`.

**Filtering:**
- Removes rows where gender or age is `NaN`.
- Converts textual age groups to numerical values.
- Removes samples labeled as 'other' gender to maintain binary classification.

**Balancing:**
- To avoid gender bias, the dataset is balanced by downsampling the overrepresented gender based on age distribution.

## Data Preprocessing

**Sampling Parameters (Default Values)**
- Sample Rate: `22050 Hz`
- Clip Duration: `5 seconds`
- Target Size: `5 * 22050` = `110250` samples

**Audio Preprocessing Pipeline**
- Audio files are loaded using `librosa.load()`.
- Each audio sample is cropped (randomly or center for test set) to 5 seconds (110,250 samples) with padding for shorter clips.
- MFCC (Mel-frequency cepstral coefficients) features are extracted using `librosa.feature.mfcc()`
- Features are standardized using mean and standard deviation to improve convergence during training.

**Dataset Creation**

- TensorFlow Dataset API is used to create efficient data pipelines using `tf.data.Dataset`
- Batches of `256` samples are created
- Training data is shuffled and cached for performance
- Test data uses center cropping instead of random cropping for consistency

## Model Architecture

The model combines CNN and RNN architectures for effective audio feature extraction:

**Input Layer**
- Input shape `(n_mfcc, time)` is determined by the MFCC features (varies based on audio length and feature extraction parameters)

**Convolutional Layers**
- First Conv1D: `32` filters with `3×1` kernel, stride `1`
- Second Conv1D: `64` filters with `3×1` kernel, stride `1`
- Third Conv1D: `128` filters with `3×1` kernel, stride `1`
- Each convolutional layer is followed by **batch normalization** and **ReLU** activation

**Recurrent Layer**
- **GRU** layer with `128` units processes temporal information in the sequence

**Output Layers**
- Flatten layer to convert features to 1D
- Gender output: Dense layer with **sigmoid** activation (binary classification)
- Age output: Dense layer with **linear** activation (regression)

## Model Training

The training approach uses a multi-task learning setup:

**Loss Functions**
- Gender prediction: **Binary Cross-Entropy** loss (classification task)
- Age prediction: **Mean Squared Logarithmic (MSLE)** Error (regression task). Suitable due to age being positive, wide-ranged and possibly skewed
- Loss weights: Equal weighting (0.5 for each task), balances influence from classification and regression

**Optimizer**
- **Adam Optimizer** with **learning_rate** = `0.0001`.
- Chosen for adaptive learning and stability in noisy gradients common in audio data

**Metrics:**
- **Accuracy** for gender
- **MeanAbsoluteError** for age — interpretable in years

**Training Setup:**
- **Epochs**: `30`
- **Batch Size**: `256`
- **Steps Per Epoch**: Computed from dataset size
- **Prefetch and Cache**: Ensures efficient GPU/TPU usage during training
- Model performance is validated on a separate test set
- Validation is performed after each epoch
