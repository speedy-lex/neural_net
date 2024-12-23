pub mod network;
pub mod activations;
pub mod data;
pub mod mnist;
pub mod gradient;
pub mod image;
pub mod cost;

use activations::Activation::*;

use data::elemax;

const NEW_NETWORK: bool = false;
const TRAIN: bool = false;

fn main() {
    println!("loading/creating network");
    let mut net: network::Network;
    let mut src = random::default(3201);

    if NEW_NETWORK {
        net = network::Network::new_rand(vec!(28*28, 100, 10), vec![Sigmoid, Sigmoid], cost::CostFunction::Mse, &mut src);
    } else {
        net = network::Network::from_file("network.bin");
    }

    println!("image.png is {}", elemax(net.evaluate(&image::get())));
    let mut el = net.evaluate(&image::get()).into_iter().enumerate().collect::<Vec<(usize, f64)>>();
    el.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (num, prob) in el.iter().rev() {
        println!("{num}: {}%", (prob*10000.0) as usize as f64 / 100.0);
    }

    if TRAIN {
        let (data_test, test_labels) = mnist::get_mnist_test();
        let batch_size = 100;
        for _ in 0..16 {
            println!("generating batches");
            let (data, expected) = mnist::get_mnist_train(&mut src);
            let (batches, labels_batched) = data::get_mini_batch(data.clone(), expected.clone(), batch_size);
            println!("training");
            net = net.learn_epoch(&batches, &labels_batched, 1.0);
            println!("testing");
            println!("{}% training data", net.accuracy(&data, &expected)*100.0);
            println!("{}% test data", net.accuracy(&data_test, &test_labels)*100.0);
            println!("saving_model");
            net.save_to_file("network.bin");
        }
    }
}