use std::time::Instant;

use crate::SIZE;

extern "C" {
    pub fn blackbox(dummy: f32) -> f32;
}

pub fn run() {
    let start = Instant::now();

    let data = vec![1f32; SIZE];

    let mut prefix_sum = vec![0f32; SIZE];

    prefix_sum[0] = data[0];
    for i in 1..data.len() {
        unsafe {
            prefix_sum[i] = blackbox(prefix_sum[i - 1]) + data[i];
        }
    }

    println!("the total value is {:?}", prefix_sum.last().unwrap());

    println!("calculating that took {:?}", start.elapsed());
}
