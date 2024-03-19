use std::ops::{Add, AddAssign};

#[derive(Clone)]
pub struct Gradients {
    pub biases: Vec<Vec<f64>>,
    pub weights: Vec<Vec<Vec<f64>>>
}
impl Gradients {
    pub fn new(shape: &Vec<usize>) -> Self {
        let mut x = Self{biases: vec![], weights: vec![]};
        for i in 1..shape.len() {
            x.biases.push(vec![0.0; shape[i]]);
            x.weights.push(vec![vec![0.0; shape[i-1]]; shape[i]]);
        }
        x
    }
    pub fn print(&self) {
        println!("{:#?}", self.biases);
        println!("{:#?}", self.weights);
    }
}
impl Add for Gradients {
    type Output=Gradients;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new = self.clone();
        for x in 0..new.biases.len() {
            for y in 0..new.biases[x].len() {
                new.biases[x][y] = self.biases[x][y] + rhs.biases[x][y];
            }
        }
        for x in 0..new.weights.len() {
            for y in 0..new.weights[x].len() {
                for z in 0..new.weights[x][y].len() {
                    new.weights[x][y][z] = self.weights[x][y][z] + rhs.weights[x][y][z];
                }
            }
        }
        new
    }
}
impl AddAssign for Gradients {
    fn add_assign(&mut self, rhs: Self) {
        for x in 0..self.biases.len() {
            for y in 0..self.biases[x].len() {
                self.biases[x][y] += rhs.biases[x][y];
            }
        }
        for x in 0..self.weights.len() {
            for y in 0..self.weights[x].len() {
                for z in 0..self.weights[x][y].len() {
                    self.weights[x][y][z] += rhs.weights[x][y][z];
                }
            }
        }
    }
}

#[test]
fn grad_test() {
    Gradients::new(&vec![1, 2, 1]).print()
}