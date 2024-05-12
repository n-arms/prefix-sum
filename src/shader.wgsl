@group(0)
@binding(0)
var<storage, read_write> data: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> sums: array<f32>;


const wgsize: u32 = 4;
const n: u32 = wgsize * 2;

var<workgroup> shared_data: array<f32, n>;

@compute
@workgroup_size(4)
fn scan_main(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let start = workgroup_id.x * n;
    let tid = local_id.x;
    var offset: u32 = 1;
    shared_data[2*tid] = data[start + 2*tid];
    shared_data[2*tid + 1] = data[start + 2*tid + 1];

    for (var d: u32 = wgsize; d > 0; d = d >> 1) {
        workgroupBarrier();
        if (tid < d) {
            let a = offset*(2*tid + 1) - 1;
            let b = offset*(2*tid + 2) - 1;
            shared_data[b] += shared_data[a];
        }
        offset = offset * 2;
    }

    if (tid == 0) {
        sums[workgroup_id.x] = shared_data[n - 1];
        shared_data[n - 1] = 0.0;
    }

    for (var d: u32 = 1; d < n; d = d * 2) {
        offset = offset / 2;
        workgroupBarrier();
        if (tid < d) {
            let a = offset*(2*tid + 1) - 1;
            let b = offset*(2*tid + 2) - 1;
            let temp = shared_data[a];
            shared_data[a] = shared_data[b];
            shared_data[b] += temp;
        }
    }
    workgroupBarrier();

    data[start + 2*tid] = shared_data[2*tid];
    data[start + 2*tid + 1] = shared_data[2*tid + 1];
}

@compute
@workgroup_size(4)
fn add_main(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let to_add = sums[workgroup_id.x];
    let a = workgroup_id.x * n + local_id.x * 2;
    let b = workgroup_id.x * n + local_id.x * 2 + 1;

    data[a] = data[a] + to_add;
    data[b] = data[b] + to_add;
}