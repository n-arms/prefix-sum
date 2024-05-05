use std::time::Instant;

use crate::SIZE;

pub fn run() {
    let start = Instant::now();

    let data = vec![1f32; SIZE];

    let mut prefix_sum = vec![0f32; SIZE];

    prefix_sum[0] = data[0];
    for i in 1..data.len() {
        prefix_sum[i] = prefix_sum[i - 1] + data[i];
    }

    println!("the total value is {:?}", prefix_sum.last().unwrap());

    println!("calculating that took {:?}", start.elapsed());
}
