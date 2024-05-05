@group(0)
@binding(0)
var<storage, read_write> buffer: array<f32>;

@group(0)
@binding(1)
var<uniform> stride: u32;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let start = global_id.x * stride;

    let existing_sum = buffer[start + stride / 2 - 1];
    
    for (var i: u32 = stride / 2; i < stride; i = i + 1) {
        buffer[start + i] = buffer[start + i] + existing_sum;
    }
}