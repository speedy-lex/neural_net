use std::f64::consts::E;

#[derive(serde::Serialize, serde::Deserialize, Clone, Copy)]
pub enum Activation {
    Relu,
    Sigmoid,
    Zero,
}
impl Activation {
    pub fn activate(&self, x: f64) -> f64 {
        match *self {
            Self::Relu => {
                relu(x)
            }
            Self::Sigmoid => {
                sigmoid(x)
            }
            Self::Zero => {
                x
            }
        }
    }
    pub fn derivate(&self, x: f64) -> f64 {
        match *self {
            Self::Relu => {
                relu_derivative(x)
            }
            Self::Sigmoid => {
                sigmoid_derivative(x)
            }
            Self::Zero => {
                1.0
            }
        }
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}
fn relu_derivative(x: f64) -> f64 {
    if x>0.0 {
        1.0
    } else {
        0.0
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0/(1.0+E.powf(-x))
}
fn sigmoid_derivative(x: f64) -> f64 {
    let sigm = sigmoid(x);
    sigm * (1.0-sigm)
}