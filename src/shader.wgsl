@group(0)
@binding(0)
var<storage, read_write> input: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> output: array<f32>;


const wgsize: u32 = 256;

var<workgroup> shared_data: array<f32, wgsize>;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let i = workgroup_id.x * 2 * wgsize + tid;

    shared_data[tid] = input[i] + input[i + wgsize];


    workgroupBarrier();

    for (var s: u32 = wgsize / 2; s > 0; s >>= 1u) {
        if tid < s {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    if tid == 0 {
        output[global_id.x / wgsize] = shared_data[0];

    }
}