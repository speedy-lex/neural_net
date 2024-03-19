use std::sync::{mpsc, Arc, RwLock};
use std::thread::available_parallelism;

use crossbeam::thread;
use random::Source;

use crate::activations::Activation;
use crate::cost::CostFunction;
use crate::data::elemax;
use crate::gradient::Gradients;

#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone)]
pub struct Node {
    weights: Vec<f64>,
    bias: f64,
}
impl Node {
    pub fn new(prev_layer_size: usize) -> Self {
        Self{weights: vec![0.0; prev_layer_size], bias: 0.0}
    }
    pub fn new_rand(prev_layer_size: usize, src: &mut random::Default) -> Self {
        let mut x = Self{weights: vec![], bias: src.read::<f64>()*2.0-1.0};
        for _ in 0..prev_layer_size {
            x.weights.push(src.read::<f64>()*2.0-1.0);
        }
        x
    }
    pub fn evaluate(&self, prev_layer: &Vec<f64>) -> f64 {
        let mut total = 0.0;
        for (value, weight) in prev_layer.iter().zip(self.weights.iter()) {
            total += value*weight;
        }
        total + self.bias
    }
    pub fn apply_gradients(&mut self, bias:f64, weights: &Vec<f64>, learn_rate: f64) {
        self.bias -= bias * learn_rate;
        for (weight, &gradient) in self.weights.iter_mut().zip(weights.iter()) {
            *weight -= gradient * learn_rate;
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Layer {
    pub nodes: Vec<Node>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(size: usize, prev_layer_size: usize, activation: Activation) -> Self {
        let mut x = Self{nodes: vec![], activation};
        for _ in 0..size {
            x.nodes.push(Node::new(prev_layer_size))
        }
        x
    }
    pub fn new_rand(size: usize, prev_layer_size: usize, activation: Activation, src: &mut random::Default) -> Self {
        let mut x = Self{nodes: vec![], activation};
        for _ in 0..size {
            x.nodes.push(Node::new_rand(prev_layer_size, src))
        }
        x
    }
    pub fn new_first_layer(size: usize) -> Self {
        Self {nodes: vec![Node::new(0); size], activation: Activation::Zero}
    }
    pub fn evaluate(&self, prev_layer: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut weighted_inputs = Vec::with_capacity(self.nodes.len());
        let mut results = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let node_eval = node.evaluate(prev_layer);
            weighted_inputs.push(node_eval);
            results.push(self.activation.activate(node_eval))
        }
        (results, weighted_inputs)
    }
    pub fn apply_gradients(&mut self, biases:&Vec<f64>, weights: &Vec<Vec<f64>>, learn_rate: f64) {
        for ((bias, weight), node) in biases.iter().zip(weights.iter()).zip(self.nodes.iter_mut()) {
            node.apply_gradients(*bias, weight, learn_rate)
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub epochs: usize,
    shape: Vec<usize>,
    pub cost: CostFunction,
}
impl Network {
    pub fn new(shape: Vec<usize>, activations:Vec<Activation>, cost:CostFunction) -> Self {
        let mut x = Self{layers: vec![], shape: shape.clone(), epochs: 0, cost};
        x.layers.push(Layer::new_first_layer(shape[0]));
        let mut prev = shape[0];
        for (size, activation) in shape.into_iter().skip(1).zip(activations.into_iter()) {
            x.layers.push(Layer::new(size, prev, activation));
            prev = size;
        }
        x
    }
    pub fn new_rand(shape: Vec<usize>, activations:Vec<Activation>, cost:CostFunction, src: &mut random::Default) -> Self {
        let mut x = Self{layers: vec![], shape: shape.clone(), epochs: 0, cost};
        x.layers.push(Layer::new_first_layer(shape[0]));
        let mut prev = shape[0];
        for (size, activation) in shape.into_iter().skip(1).zip(activations.into_iter()) {
            x.layers.push(Layer::new_rand(size, prev, activation, src));
            prev = size;
        }
        x
    }
    pub fn evaluate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut prev = inputs.clone();
        for x in 1..self.layers.len() {
            prev = self.layers[x].evaluate(&prev).0;
        }
        prev
    }
    pub fn evaluate_layer_results(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut prev = (inputs.clone(), vec![]);
        let mut res = vec![];
        let mut weighted = vec![];
        for x in 1..self.layers.len() {
            let x = self.layers[x].evaluate(&prev.0);
            res.push(prev.0);
            weighted.push(prev.1);
            prev = x;
        }
        res.push(prev.0.clone());
        weighted.push(prev.1);
        (prev.0, res, weighted)
    }
    pub fn cost(&self, data: &Vec<Vec<f64>>, expected: &Vec<Vec<f64>>) -> f64 {
        let mut cost = 0.0;
        for (point, expect) in data.into_iter().zip(expected.into_iter()) {
            let eval = self.evaluate(point);
            cost += self.cost.cost(&eval, expect);
        }
        cost
    }
    pub fn apply_gradients(&mut self, gradients: Gradients, learn_rate: f64) {
        for ((biases, weights), layer) in gradients.biases.iter().zip(gradients.weights.iter()).zip(self.layers.iter_mut().skip(1)) {
            layer.apply_gradients(biases, weights, learn_rate)
        }
    }
    fn back_propagate(&self, data: &Vec<f64>, expected: &Vec<f64>) -> Gradients {
        let (out, results, weighted) = self.evaluate_layer_results(data);
        let num_layers = self.layers.len();
        let mut gradient = Gradients::new(&self.shape);
        let mut node_vals=vec![vec![]; num_layers];
        
        // last layer gradients
        let prev_layer = &self.layers[num_layers-2];
        let current_layer = &self.layers[num_layers-1];
        for node_index in 0..current_layer.nodes.len() {
            node_vals.last_mut().unwrap().push(self.cost.node_derivate(out[node_index], expected[node_index]) * current_layer.activation.derivate(weighted.last().unwrap()[node_index]));
        }
        for node_index in 0..current_layer.nodes.len() {
            for prev_node_index in 0..prev_layer.nodes.len() {
                gradient.weights[num_layers-2][node_index][prev_node_index] += node_vals.last().unwrap()[node_index] * results[num_layers-3][prev_node_index];
            }
            gradient.biases[num_layers-2][node_index] += node_vals.last().unwrap()[node_index];
        }
        for layer_num in (1..num_layers-1).rev() {
            // node values
            for node_index in 0..self.layers[layer_num].nodes.len() {
                node_vals[layer_num].push(0.0);
                for next_l_node_index in 0..self.layers[layer_num+1].nodes.len() {
                    node_vals[layer_num][node_index] += node_vals[layer_num+1][next_l_node_index] * self.layers[layer_num+1].nodes[next_l_node_index].weights[node_index];
                }
                node_vals[layer_num][node_index] *= self.layers[layer_num].activation.derivate(weighted[layer_num][node_index]);
            }
            // update gradients
            for node_index in 0..self.layers[layer_num].nodes.len() {
                gradient.biases[layer_num-1][node_index] += node_vals[layer_num][node_index];
                //self.layers[layer_num].nodes[node_index].bias_gradient += node_vals[layer_num][node_index];
                for prev_l_node_index in 0..self.layers[layer_num-1].nodes.len() {
                    gradient.weights[layer_num-1][node_index][prev_l_node_index] += node_vals[layer_num][node_index] * results[layer_num-1][prev_l_node_index];
                    // self.layers[layer_num].nodes[node_index].weight_gradients[prev_l_node_index] += node_vals[layer_num][node_index] * results[layer_num-1][prev_l_node_index];
                }
            }
        }
        gradient
    }
    pub fn learn_single(&mut self, data: &Vec<f64>, expected: &Vec<f64>, learn_rate: f64) {
        self.apply_gradients(self.back_propagate(data, expected), learn_rate)
    }
    pub fn learn_batch(&mut self, data: &Vec<Vec<f64>>, expected: &Vec<Vec<f64>>, learn_rate: f64) {
        let num_points = data.len() as f64;
        let mut v = vec![];
        for (point, output) in data.into_iter().zip(expected.into_iter()) {
            v.push(self.back_propagate(point, output));
        }
        for grad in v {
            self.apply_gradients(grad, learn_rate / num_points);
        }
    }
    pub fn learn_epoch(self, data: &Vec<Vec<Vec<f64>>>, expected: &Vec<Vec<Vec<f64>>>, learn_rate: f64) -> Self {
        crossbeam::scope(|spawner| {
            let rw = Arc::new(RwLock::new(self));
            let thread_count = available_parallelism().unwrap().get();
            let chunk_size = (data[0].len() as f64 / thread_count as f64).ceil() as usize;
            let batch_size = data[0].len();
            let batch_count = data.len();
            let mut threads = vec![];
            let mut channels: Vec<mpsc::Sender<(Vec<Vec<f64>>, Vec<Vec<f64>>)>> = vec![];
            let (ret, returns) = mpsc::channel();
            for _ in 0..thread_count {
                let r = ret.clone();
                let (sender, receiver) = mpsc::channel();
                let lock = rw.clone();
                channels.push(sender);
                threads.push(spawner.spawn(move |_| {
                    for (data, labels) in receiver {
                        
                        let mut grad = Gradients::new(&lock.read().unwrap().shape);
                        let guard = lock.read().unwrap();
                        for (dat, lab) in data.iter().zip(labels.iter()) {
                            grad+=guard.back_propagate(dat, lab);
                        }
                        r.send(grad).unwrap();
                    }
                }));
            }
            for (i, (batch_data, batch_labels)) in data.iter().zip(expected.iter()).enumerate() {
                for ((chunk_data, chunk_labels), pipe) in batch_data.chunks(chunk_size).zip(batch_labels.chunks(chunk_size)).zip(channels.iter()) {
                    pipe.send((chunk_data.to_vec(), chunk_labels.to_vec())).unwrap();
                }
                let mut grad = Gradients::new(&rw.read().unwrap().shape);
                let mut counter = 0;
                while counter < thread_count-1 {
                    grad += returns.recv().unwrap();
                    counter += 1;
                }
                if ((i+1)*20/batch_count) as f64 == (((i+1)*20) as f64/batch_count as f64) {
                    println!("{}%", (i+1)*100/batch_count);
                }
                rw.write().unwrap().apply_gradients(grad, learn_rate/batch_size as f64);
            }
            let mut g = rw.write().unwrap();
            g.epochs+=1;
            g.clone()
        }).unwrap()
    }
    pub fn accuracy_single_threaded(&self, data: &Vec<Vec<f64>>, expected: &Vec<Vec<f64>>) -> f64 {
        let total = data.len();
        let mut correct = 0;
        for (point, label) in data.iter().zip(expected.iter()) {
            let output = self.evaluate(point);
            let mut max = -1e6;
            let mut idx = 0;
            for (node, i) in output.into_iter().zip(0..) {
                if node > max {
                    max = node;
                    idx = i;
                }
            }
            for i in 0..label.len() {
                if label[i]>0.0 {
                    if idx == i {
                        correct += 1;
                    }
                    break;
                }
            }
        }
        correct as f64/total as f64
    }
    pub fn accuracy(&self, data: &Vec<Vec<f64>>, expected: &Vec<Vec<f64>>) -> f64 {
        thread::scope(|spawner| {
            let thread_count = available_parallelism().unwrap().get();
            let chunk_size = (data.len() as f64 / thread_count as f64).ceil() as usize;
            let mut threads = vec![];
            for (dat, labels) in data.chunks(chunk_size).zip(expected.chunks(chunk_size)) {
                let net = self;
                threads.push(spawner.spawn(|_| {
                    let mut correct = 0;
                    for (d, l) in dat.iter().zip(labels.to_owned().into_iter()) {
                        if elemax(net.evaluate(d))==elemax(l) {
                            correct += 1;
                        }
                    }
                    correct
                }));
            } 
            let mut total = 0;
            for thread in threads {
                total += thread.join().unwrap();
            }
            total
        }).unwrap() as f64 / data.len() as f64
    }
}