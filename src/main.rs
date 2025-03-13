use mnist::*;
use ndarray::{prelude::*, stack, ShapeError};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, error::Error, fs::File, io::Read, rc::Rc};

const SAVE_FILE: &str = "./data.json";

thread_local! {
    static MNIST_DATA: Rc<RefCell<Option<MnistData>>> = Rc::new(RefCell::new(None));
}

trait NeuralNetworkLayer {
    fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32>;
    fn backward(&mut self, output_gradient: &Array1<f32>, learning_rate: f32) -> Array1<f32>;
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
enum Layer {
    Dense {
        weights: Array2<f32>,
        biases: Array1<f32>,
        input: Array1<f32>,
        num_inputs: usize,
        num_outputs: usize,
    },
    Sigmoid {
        input: Array1<f32>,
    },
    Softmax {
        output: Array1<f32>,
    },
    Tanh {
        input: Array1<f32>,
    },
}

impl Layer {
    fn random_layer(num_inputs: usize, num_outputs: usize) -> Self {
        Self::Dense {
            weights: Array2::random((num_outputs, num_inputs), Uniform::new(-1.0, 1.0)),
            biases: Array1::random(num_outputs, Uniform::new(0.0, 1.0)),
            input: Array1::zeros(num_inputs),
            num_inputs,
            num_outputs,
        }
    }

    fn sigmoid() -> Self {
        Self::Sigmoid {
            input: Array1::zeros(1),
        }
    }

    fn softmax() -> Self {
        Self::Softmax {
            output: Array1::zeros(1),
        }
    }

    fn tanh() -> Self {
        Self::Tanh {
            input: Array1::zeros(1),
        }
    }
}

impl NeuralNetworkLayer for Layer {
    fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        match self {
            Self::Dense {
                weights,
                biases,
                input,
                ..
            } => {
                *input = inputs.clone();
                weights.dot(inputs) + &*biases
            }
            Self::Sigmoid { input } => {
                *input = inputs.clone();
                inputs.mapv(sigmoid)
            }
            Self::Softmax { output } => {
                let tmp = inputs.mapv(f32::exp);
                let sum = tmp.clone().sum();
                *output = tmp / sum;
                output.clone()
            }
            Self::Tanh { input } => {
                *input = inputs.clone();
                inputs.mapv(|x| x.tanh())
            }
        }
    }

    fn backward(&mut self, output_gradient: &Array1<f32>, learning_rate: f32) -> Array1<f32> {
        match self {
            Self::Dense {
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
            Self::Sigmoid { input } => {
                output_gradient
                    * input.mapv(|x| {
                        let sig = sigmoid(x);
                        sig * (1.0 - sig)
                    })
            }
            Self::Softmax { output } => {
                let n = output.len();
                let repeated = (0..n).map(|_| output.view()).collect::<Vec<_>>();
                let tiled = stack(Axis(0), &repeated).unwrap().mapv(|x| x);
                let identity =
                    Array2::<f32>::from_shape_fn((n, n), |(i, j)| if i == j { 1.0 } else { 0.0 });
                (tiled.clone() * (identity - tiled.t())).dot(output_gradient)
            }
            Self::Tanh { input } => output_gradient * input.mapv(|x| 1.0 - x.tanh().powi(2)),
        }
    }
}

impl From<Layer> for LayerData {
    fn from(layer: Layer) -> LayerData {
        match layer {
            Layer::Sigmoid { .. } => LayerData::Sigmoid,
            Layer::Softmax { .. } => LayerData::Softmax,
            Layer::Tanh { .. } => LayerData::Tanh,
            Layer::Dense {
                weights,
                biases,
                num_inputs,
                num_outputs,
                ..
            } => LayerData::DenseLayer {
                weights: weights.into_raw_vec_and_offset().0,
                biases: biases.to_vec(),
                num_inputs,
                num_outputs,
            },
        }
    }
}

fn mse(output: &Array1<f32>, expected_output: &Array1<f32>) -> f32 {
    (expected_output - output).mapv(|x| x * x).sum() / output.shape()[0] as f32
}

fn mse_prime(output: &Array1<f32>, expected_output: &Array1<f32>) -> Array1<f32> {
    (output - expected_output).mapv(|x| 2.0 * x / expected_output.len() as f32)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum LayerData {
    Sigmoid,
    Softmax,
    Tanh,
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
            LayerData::Sigmoid => Layer::sigmoid(),
            LayerData::Softmax => Layer::softmax(),
            LayerData::Tanh => Layer::tanh(),
            LayerData::DenseLayer {
                weights,
                biases,
                num_inputs,
                num_outputs,
            } => Layer::Dense {
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
}

#[derive(Debug, Clone)]
struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        let mut output = input;
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    fn train(
        &mut self,
        inputs: &Array2<f32>,
        expected_outputs: &Array2<f32>,
        num_epochs: usize,
        learning_rate: f32,
    ) {
        for _ in 0..num_epochs {
            let mut error = 0.0;
            for (input, expected_output) in inputs
                .clone()
                .rows()
                .into_iter()
                .zip(expected_outputs.clone().rows())
            {
                let input = input.mapv(|x| x);
                let expected_output = expected_output.mapv(|x| x);
                let output = self.forward(input);
                error += mse(&output, &expected_output);
                let mut gradient = mse_prime(&output, &expected_output);
                for layer in self.layers.iter_mut().rev() {
                    gradient = layer.backward(&gradient, learning_rate);
                }
            }

            println!("Error: {:.8}", error / inputs.shape()[0] as f32,);
        }
    }

    fn test(&mut self, inputs: &Array2<f32>, expected_outputs: &Array2<f32>) -> f32 {
        let mut wrong: usize = 0;
        let total = inputs.clone().shape()[0];
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
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let expected_answer = expected_output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if answer != expected_answer {
                wrong += 1;
            }
        }
        1.0 - (wrong as f32 / total as f32)
    }

    fn save(&self) -> Result<(), Box<dyn Error>> {
        let data: NeuralNetworkData = self.into();
        Ok(std::fs::write(SAVE_FILE, serde_json::to_string(&data)?)?)
    }

    fn load() -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(SAVE_FILE)?;
        let mut string = String::new();
        file.read_to_string(&mut string)?;
        let data: NeuralNetworkData = serde_json::from_str(&string)?;
        Ok(data.try_into()?)
    }
}

impl From<&NeuralNetwork> for NeuralNetworkData {
    fn from(value: &NeuralNetwork) -> Self {
        NeuralNetworkData {
            layers: value.layers.clone().into_iter().map(|x| x.into()).collect(),
        }
    }
}

impl TryFrom<NeuralNetworkData> for NeuralNetwork {
    type Error = ShapeError;
    fn try_from(value: NeuralNetworkData) -> Result<Self, Self::Error> {
        let layers: Result<Vec<Layer>, ShapeError> =
            value.layers.into_iter().map(|x| x.try_into()).collect();
        Ok(NeuralNetwork { layers: layers? })
    }
}

fn get_trained_model() -> NeuralNetwork {
    match NeuralNetwork::load() {
        Ok(neural_network) => neural_network,
        Err(_) => {
            let mnist_data = load_mnist_data().unwrap();
            let mut neural_network = NeuralNetwork {
                layers: vec![
                    Layer::random_layer(28 * 28, 64),
                    Layer::tanh(),
                    Layer::random_layer(64, 16),
                    Layer::tanh(),
                    Layer::random_layer(16, 10),
                    Layer::softmax(),
                ],
            };
            neural_network.train(
                &mnist_data.train_images,
                &mnist_data.train_labels,
                1000,
                0.005,
            );
            let _ = neural_network.save();
            neural_network
        }
    }
}

fn main() {
    let mnist_data = load_mnist_data().unwrap();
    let mut model = get_trained_model();
    let accuracy = model.test(&mnist_data.test_images, &mnist_data.test_labels);
    println!("accuracy: {:.2}%", accuracy * 100.0);
}

#[derive(Debug, Clone)]
struct MnistData {
    train_images: Array2<f32>,
    train_labels: Array2<f32>,
    test_images: Array2<f32>,
    test_labels: Array2<f32>,
}

fn load_mnist_data() -> Result<MnistData, ShapeError> {
    MNIST_DATA.with(|data| {
        if let Some(data) = &*data.clone().borrow() {
            return Ok(data.clone());
        }
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
        let new_data = MnistData {
            train_images: Array2::from_shape_vec((training_sample, 28usize * 28), trn_img)?
                .mapv(|x| x as f32),
            train_labels: Array2::from_shape_vec((training_sample, 10usize), trn_lbl)?
                .mapv(|x| x as f32),
            test_images: Array2::from_shape_vec((10_000, 28 * 28), tst_img)?.mapv(|x| x as f32),
            test_labels: Array2::from_shape_vec((10_000, 10), tst_lbl)?.mapv(|x| x as f32),
        };
        *data.borrow_mut() = Some(new_data.clone());
        Ok(new_data)
    })
}
