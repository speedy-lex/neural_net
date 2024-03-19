pub fn get_mini_batch(data: Vec<Vec<f64>>, expected: Vec<Vec<f64>>, size: usize) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) {
    let batches = data.chunks(size).map(|x| x.to_vec()).collect();
    let expects = expected.chunks(size).map(|x| x.to_vec()).collect();
    (batches, expects)
}
pub fn elemax(out: Vec<f64>) -> usize {
    out.iter().enumerate().fold((0, 0.0), |(idx, val), (i, v)| {
        if val > *v {
            (idx, val)
        } else {
            (i, *v)
        }
    }).0
}

#[test]
fn elemax_test() {
    assert_eq!(elemax(vec![0.0, 0.0, 0.2, 1.0, 0.0, 0.0]), 3)
}