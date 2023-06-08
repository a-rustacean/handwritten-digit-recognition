extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use std::{error::Error, fs::File, io::Read};

use mnist::*;
use ndarray::{prelude::*, ShapeError};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

trait NeuralNetworkLayer {
    fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32>;
    fn backward(&mut self, output_gradient: &Array1<f32>, learning_rate: f32) -> Array1<f32>;
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
enum Layer {
    DenseLayer {
        weights: Array2<f32>,
        biases: Array1<f32>,
        input: Array1<f32>,
        num_inputs: usize,
        num_outputs: usize,
    },
    Sigmoid,
}

impl Layer {
    fn random_layer(num_inputs: usize, num_outputs: usize) -> Self {
        Self::DenseLayer {
            weights: Array2::random((num_outputs, num_inputs), Uniform::new(-1.0, 1.0)),
            biases: Array1::random(num_outputs, Uniform::new(0.0, 1.0)),
            input: Array1::zeros(num_inputs),
            num_inputs,
            num_outputs,
        }
    }
}

impl NeuralNetworkLayer for Layer {
    fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        match self {
            Self::DenseLayer {
                weights,
                biases,
                input,
                ..
            } => {
                *input = inputs.clone();
                weights.dot(inputs) + &*biases
            }
            Self::Sigmoid => inputs.mapv(sigmoid),
        }
    }

    fn backward(&mut self, output_gradient: &Array1<f32>, learning_rate: f32) -> Array1<f32> {
        match self {
            Self::DenseLayer {
                weights,
                biases,
                input,
                num_inputs,
                num_outputs,
            } => {
                let weight_gradient_raw: Vec<f32> = output_gradient
                    .iter()
                    .flat_map(|a| input.iter().map(|b| *a * b))
                    .collect();
                let weight_gradient =
                    Array2::from_shape_vec((*num_outputs, *num_inputs), weight_gradient_raw)
                        .unwrap();
                *weights = &*weights - weight_gradient.mapv(|x| x * learning_rate);
                *biases = &*biases - output_gradient.mapv(|x| x * learning_rate);
                weights.t().dot(output_gradient)
            }
            Self::Sigmoid => output_gradient.mapv(|x| {
                let sig = sigmoid(x);
                sig * (1.0 - sig)
            }),
        }
    }
}

impl From<Layer> for LayerData {
    fn from(layer: Layer) -> LayerData {
        match layer {
            Layer::Sigmoid => LayerData::Sigmoid,
            Layer::DenseLayer {
                weights,
                biases,
                num_inputs,
                num_outputs,
                ..
            } => LayerData::DenseLayer {
                weights: weights.into_raw_vec(),
                biases: biases.to_vec(),
                num_inputs,
                num_outputs,
            },
        }
    }
}

fn mse(output: &Array1<f32>, expected_output: &Array1<f32>) -> f32 {
    (expected_output - output).mapv(|x| x * x).sum()
}

fn mse_prime(output: &Array1<f32>, expected_output: &Array1<f32>) -> Array1<f32> {
    (expected_output - output).mapv(|x| 2.0 * x / expected_output.len() as f32)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum LayerData {
    Sigmoid,
    DenseLayer {
        weights: Vec<f32>,
        biases: Vec<f32>,
        num_inputs: usize,
        num_outputs: usize,
    },
}

impl TryFrom<LayerData> for Layer {
    type Error = ShapeError;
    fn try_from(value: LayerData) -> Result<Self, Self::Error> {
        let layer = match value {
            LayerData::Sigmoid => Layer::Sigmoid,
            LayerData::DenseLayer {
                weights,
                biases,
                num_inputs,
                num_outputs,
            } => Layer::DenseLayer {
                weights: Array2::from_shape_vec((num_outputs, num_inputs), weights)?,
                biases: Array1::from_vec(biases),
                input: Array1::zeros(num_inputs),
                num_inputs,
                num_outputs,
            },
        };
        Ok(layer)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct NeuralNetworkData {
    layers: Vec<LayerData>,
    inputs: Vec<f32>,
    expected_outputs: Vec<f32>,
    num_epochs: usize,
    learning_rate: f32,
    input_shape: (usize, usize),
    expected_output_shape: (usize, usize),
}

struct NeuralNetwork {
    layers: Vec<Layer>,
    inputs: Array2<f32>,
    expected_outputs: Array2<f32>,
    num_epochs: usize,
    learning_rate: f32,
}

impl NeuralNetwork {
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        let mut output = input;
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    fn train(&mut self) {
        for _ in 0..self.num_epochs {
            let mut error = 0.0;
            for (input, expected_output) in self
                .inputs
                .clone()
                .rows()
                .into_iter()
                .zip(self.expected_outputs.clone().rows())
            {
                let input = input.mapv(|x| x);
                let expected_output = expected_output.mapv(|x| x);
                let output = self.forward(input);
                error += mse(&output, &expected_output);
                let mut gradient = mse_prime(&output, &expected_output);
                for layer in self.layers.iter_mut().rev() {
                    gradient = layer.backward(&gradient, self.learning_rate);
                }
            }
            println!("Error: {}", error / self.inputs.shape()[0] as f32);
        }
    }

    fn test(&mut self, inputs: &Array2<f32>, expected_outputs: &Array2<f32>) -> f32 {
        let mut wrong: usize = 0;
        for (input, expected_output) in inputs
            .clone()
            .rows()
            .into_iter()
            .zip(expected_outputs.clone().rows())
        {
            let input = input.mapv(|x| x);
            let expected_output = expected_output.mapv(|x| x);
            let output = self.forward(input.clone());
            let answer = output
                .to_vec()
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max);
            let expected_answer = expected_output
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max);
            if answer != expected_answer {
                wrong += 1;
            }
        }
        1.0 - (wrong as f32 / self.inputs.len() as f32)
    }

    fn save(&self) -> Result<(), Box<dyn Error>> {
        let data: NeuralNetworkData = self.into();
        Ok(std::fs::write(
            "./data.json",
            serde_json::to_string(&data)?,
        )?)
    }

    fn load() -> Result<Self, Box<dyn Error>> {
        let mut file = File::open("./data.json")?;
        let mut string = String::new();
        file.read_to_string(&mut string)?;
        let data: NeuralNetworkData = serde_json::from_str(&string)?;
        Ok(data.try_into()?)
    }
}

impl From<&NeuralNetwork> for NeuralNetworkData {
    fn from(value: &NeuralNetwork) -> Self {
        let input_shape = value.inputs.shape();
        let input_shape = (input_shape[0], input_shape[1]);
        let expected_output_shape = value.expected_outputs.shape();
        let expected_output_shape = (expected_output_shape[0], expected_output_shape[1]);
        NeuralNetworkData {
            layers: value.layers.clone().into_iter().map(|x| x.into()).collect(),
            inputs: value.inputs.clone().into_raw_vec(),
            expected_outputs: value.expected_outputs.clone().into_raw_vec(),
            num_epochs: value.num_epochs,
            input_shape,
            expected_output_shape,
            learning_rate: value.learning_rate,
        }
    }
}

impl TryFrom<NeuralNetworkData> for NeuralNetwork {
    type Error = ShapeError;
    fn try_from(value: NeuralNetworkData) -> Result<Self, Self::Error> {
        let layers: Result<Vec<Layer>, ShapeError> =
            value.layers.into_iter().map(|x| x.try_into()).collect();
        Ok(NeuralNetwork {
            layers: layers?,
            inputs: Array2::from_shape_vec(value.input_shape, value.inputs)?,
            expected_outputs: Array2::from_shape_vec(
                value.expected_output_shape,
                value.expected_outputs,
            )?,
            num_epochs: value.num_epochs,
            learning_rate: value.learning_rate,
        })
    }
}

fn get_trained_model() -> NeuralNetwork {
    match NeuralNetwork::load() {
        Ok(neural_network) => neural_network,
        Err(_) => {
            let mnist_data = load_mnist_data().unwrap();
            let mut neural_network = NeuralNetwork {
                layers: vec![
                    Layer::random_layer(28 * 28, 16),
                    Layer::Sigmoid,
                    Layer::random_layer(16, 16),
                    Layer::Sigmoid,
                    Layer::random_layer(16, 10),
                    Layer::Sigmoid,
                ],
                inputs: mnist_data.train_images,
                expected_outputs: mnist_data.train_labels,
                num_epochs: 10,
                learning_rate: 0.0001,
            };
            neural_network.train();
            let _ = neural_network.save();
            neural_network
        }
    }
}

fn main() {
    let mnist_data = load_mnist_data().unwrap();
    let mut model = get_trained_model();
    let accuracy = model.test(&mnist_data.test_images, &mnist_data.test_labels);
    println!("Accuracy: {:.2}", accuracy * 100.0);
}

struct MnistData {
    train_images: Array2<f32>,
    train_labels: Array2<f32>,
    test_images: Array2<f32>,
    test_labels: Array2<f32>,
}

fn load_mnist_data() -> Result<MnistData, ShapeError> {
    let training_sample: usize = 50_000;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(training_sample as u32)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    Ok(MnistData {
        train_images: Array2::from_shape_vec((training_sample, 28usize * 28), trn_img)?
            .mapv(|x| x as f32),
        train_labels: Array2::from_shape_vec((training_sample, 10usize), trn_lbl)?
            .mapv(|x| x as f32),
        test_images: Array2::from_shape_vec((10_000, 28 * 28), tst_img)?.mapv(|x| x as f32),
        test_labels: Array2::from_shape_vec((10_000, 10), tst_lbl)?.mapv(|x| x as f32),
    })
}
