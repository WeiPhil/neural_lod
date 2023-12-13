#pragma once
#ifndef OPTIM_CONFIG_H
#define OPTIM_CONFIG_H

#define OPTIM_CONFIG_NAMESPACE_BEGIN namespace optimconfig {
#define OPTIM_CONFIG_NAMESPACE_END }

OPTIM_CONFIG_NAMESPACE_BEGIN


/******** Losses *******/

constexpr const char *loss_types[] = {"RelativeL2",
                                      "L2",
                                      "RelativeL1",
                                      "L1",
                                      "RelativeL2Luminance",
                                      "MAPE",
                                      "SMAPE",
                                      "CrossEntropy",
                                      "BinaryCrossEntropy",
                                      "SegmentBinaryCrossEntropy",
                                      "Hinge"};

enum LossType {
    RelativeL2,
    L2,
    RelativeL1,
    L1,
    RelativeL2Luminance,
    MAPE,
    SMAPE,
    CrossEntropy,
    BinaryCrossEntropy,
    SegmentBinaryCrossEntropy,
    Hinge,
};

/******** Encodings *******/

constexpr const char *encoding_types[] = {
    "Identity", "OneBlob", "HashGrid","TiledGrid", "DenseGrid", "Frequency", "TriangleWave", "SphericalHarmonics"};

enum EncodingType {
    Identity,
    OneBlob,
    HashGrid,
    TiledGrid,
    DenseGrid,
    Frequency,
    TriangleWave,
    SphericalHarmonics,
};

constexpr const char *interpolation_types[] = {"Nearest", "Linear", "Smoothstep"};

enum InterpolationType {
    Nearest,
    Linear,
    Smoothstep,
};

struct GridOptions{
    int n_levels = NUM_LODS();
    int n_features_per_level = 2;
    int log2_hashmap_size = 19;
    int base_resolution = VOXEL_GRID_MIN_RES();
    float per_level_scale = 2.0;
    InterpolationType interpolation = InterpolationType::Linear;
    bool stochastic_interpolation = false;
};


struct ShOptions{
    int max_degree = 8;
};

struct OptimizerOptions {
    float learning_rate = 1e-2;
    int decay_start = 10000;
    int decay_interval = 2000;
    float decay_base = 0.33;
    float beta1 = 0.9;           // Beta1 parameter of Adam.
    float beta2 = 0.999;         // Beta2 parameter of Adam.
    float epsilon = 1e-8;        // Epsilon parameter of Adam.
    float l2_reg = 1e-8;         // Strength of L2 regularization
                                 // applied to the to-be-optimized params.
    float relative_decay = 0.0;  // Percentage of weights lost per step.
    float absolute_decay = 0.0;  // Amount of weights lost per step.
    bool adabound = false;       // Whether to enable AdaBound.
};

/******** Optimizers *******/

constexpr const char *optimizer_types[] = {"SGD", "Adam", "Shampoo", "Novograd"};

enum OptimizerType {
    SGD,
    Adam,
    Shampoo,
    Novograd,
};

/******** Networks *******/

constexpr const char *network_types[] = {"FullyFusedMLP", "CutlassMLP"};

enum NetworkType {
    FullyFusedMLP,
    CutlassMLP,
};

/******** Activations *******/

constexpr const char *activation_types[] = {
    "ReLU", "Exponential", "Sigmoid", "Squareplus", "Softplus","Tanh","None"};

enum ActivationType {
    ReLU,
    Exponential,
	Sigmoid,
	Squareplus,
	Softplus,
	Tanh,
    None,
};

OPTIM_CONFIG_NAMESPACE_END

#endif