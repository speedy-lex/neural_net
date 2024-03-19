use std::fs;

use random::Source;

pub fn get_mnist_train(src: &mut random::Default) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut data = Vec::with_capacity(60_000);
    let mut labels = Vec::with_capacity(60_000);
    let mut n = 0;
    for x in fs::read_to_string("mnist_train.csv").unwrap().split('\n') {
        labels.push(vec![0.0; 10]);
        labels.last_mut().unwrap()[x[0..1].parse::<usize>().unwrap()] = 1.0;
        data.push(Vec::with_capacity(28*28));
        for pixel in x.split(',').skip(1) {
            data.last_mut().unwrap().push((pixel.parse::<u8>().unwrap() as f64)/255.0);
        }
        let clone = data.last().unwrap().clone();
        *data.last_mut().unwrap()=transform(clone, src.read_f64()/2.0-0.25, (src.read_f64()-0.5)*0.3+1.05, (src.read_f64()*6.0-3.0, src.read_f64()*6.0-3.0));
        n+=1;
        if n % 20000==0 {
            println!("{}% loading mnist", n/600)
        }
    }
    println!("loaded mnist!");
    (data, labels)
}
pub fn get_mnist_test() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut data = Vec::with_capacity(10_000);
    let mut labels = Vec::with_capacity(10_000);
    let mut n = 0;
    for x in fs::read_to_string("mnist_test.csv").unwrap().split('\n') {
        labels.push(vec![0.0; 10]);
        labels.last_mut().unwrap()[x[0..1].parse::<usize>().unwrap()] = 1.0;
        data.push(Vec::with_capacity(28*28));
        for pixel in x.split(',').skip(1) {
            data.last_mut().unwrap().push((pixel.parse::<u8>().unwrap() as f64)/255.0);
        }
        n+=1;
        if n % 2000==0 {
            println!("{}% loading mnist test", n/100)
        }
    }
    println!("loaded mnist test!");
    (data, labels)
}
pub fn translate(x: &mut Vec<f64>, src: &mut random::Default) {
    x.rotate_left(28*2+2);
    x.rotate_right((src.read_u64()%5) as usize + 28 * (src.read_u64()%5) as usize);
}
pub fn transform(img: Vec<f64>, rot:f64, scale:f64, translation: (f64, f64)) -> Vec<f64> {
    let (sin, cos) = rot.sin_cos();
    let mut i = vec![0.0; 28*28];
    for x in 0..(28_usize.pow(2)) {
        let (mut u, mut v) = ((x%28) as f64 - 14.0, (x/28) as f64 - 14.0);
        (u, v) = (u*cos-v*sin, u*sin+v*cos);
        (u, v) = (u/scale, v/scale);
        (u, v) = (u-translation.0+14.0, v-translation.1+14.0);
        let (u_frac, v_frac) = (u%1.0, v%1.0);
        let (u, v) = (u as usize, v as usize);
        let uv_lin = u+v*28;
        let (pix00, pix10, pix01, pix11) = (sample(&img, uv_lin), sample(&img, uv_lin+1), sample(&img, uv_lin+28), sample(&img, uv_lin+29));
        let (pix0, pix1) = (pix00*(1.0-v_frac)+pix01*v_frac, pix10*(1.0-v_frac)+pix11*v_frac);
        let pix = pix0*(1.0-u_frac)+pix1*u_frac;
        i[x]=pix;
    }
    i
}
fn sample(img: &Vec<f64>, idx: usize) -> f64 {
    if idx<img.len() {
        img[idx]
    } else {
        0.0
    }
}

#[test]
fn transform_test() {
    use std::{f64::consts::PI, io::Write};
    
    let img = get_mnist_test().0[0].clone();
    let img = transform(img, PI, 0.8, (2.0, 0.0));
    let img: Vec<_> = img.into_iter().map(|x|{format!("{}\n",((x*255.0) as u8))}).collect();

    let mut buf = "P2\n28 28\n255\n".to_owned();

    for x in img {
        buf+=&x;
    }

    let mut file = fs::File::create(".\\test.ppm").unwrap();
    file.write_all(buf.as_bytes()).unwrap();
}