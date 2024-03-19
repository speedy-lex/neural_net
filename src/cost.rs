#[derive(serde::Serialize, serde::Deserialize, Clone, Copy)]
pub enum CostFunction {
    Mse,
    CrossEntropy
}
impl CostFunction {
    pub fn cost(&self, predicted: &Vec<f64>, expected: &Vec<f64>) -> f64 {
        match *self {
            Self::Mse => {
                let mut cost = 0.0;
                for (&d_actual, &predicted) in predicted.iter().zip(expected.iter()) {
                    cost += (d_actual-predicted).powi(2);
                }
                cost
            }
            Self::CrossEntropy => {
                // softmax(predicted).iter().enumerate().fold(0.0, |l, (i, element)| {
                //     l-(expected[i]*element.ln())
                // });
                let softmax_predicted = softmax(predicted);
                -expected.iter().zip(softmax_predicted.iter()).map(|(a, b)| a * b.ln()).sum::<f64>()
            }
        }
    }
    pub fn node_derivate(&self, result: f64, expected: f64) -> f64 {
        match *self {
            Self::Mse => {
                2.0 * (result - expected)
            }
            Self::CrossEntropy => {
                if result == 0.0 || result == 1.0 {
                    return 0.0;
                }
                (expected - result) / (result * (result - 1.0))
            }
        }
    }
}

pub fn softmax(vals: &Vec<f64>) -> Vec<f64> {
    let tot = vals.iter().fold(0.0, |x, y|{x+y.exp()});
    vals.into_iter().map(|x|{x.exp()/tot}).collect()
}

#[test]
fn softmax_test() {
    assert_eq!(softmax(&vec![2.0, 4.0, 5.0, 3.0]).into_iter().fold(0.0, |x, y| {x+y}), 1.0);
}