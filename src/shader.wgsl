struct WorkGroupState {
    // 0: no work has been done, 1: the aggregate field contains the aggregate, 2: the aggregate is 0, 3: the inclusive prefix field contains the inclusive prefix, 4: the inclusive prefix is 0
    state: atomic<u32>,
    aggregate: atomic<u32>,
    inclusive_prefix: atomic<u32>,
    debug_info: array<u32, 256>,
    next_byte: u32
}

@group(0)
@binding(0)
var<storage, read_write> data: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> workgroup_state: array<WorkGroupState>;

@group(0)
@binding(2)
var<storage, read_write> next_block: atomic<u32>;


const wgsize: u32 = 2;
const n: u32 = wgsize * 2;

var<workgroup> shared_data: array<f32, n>;

// do a reduction over the given block, stopping each loop iteration to check if the work has already been done
fn reduce(block: u32, tid: u32) {
    let start = n * block;
    let a = data[start + tid];
    let b = data[start + tid + wgsize];
    shared_data[tid] = a + b;

    for (var d: u32 = wgsize >> 1; d > 0; d = d >> 1) {
        if (tid < d) {
            let a = shared_data[tid];
            let b = shared_data[tid + d];
            shared_data[tid] = a + b;
        }

        if (tid == 0 && block_finished(block)) {
            return;
        }
    }

    if (tid == 0) {
        let value = shared_data[0];
        let state = workgroup_state[block].state;    
        if (state != 0) { return; }
        if (value == 0) {
            workgroup_state[block].state = 2u;
        } else {
            workgroup_state[block].state = 1u;
            workgroup_state[block].aggregate = bitcast<u32>(value);
        }
    }
}

fn block_finished(block: u32) -> bool {
    let state = workgroup_state[block].state;
    let aggregate = bitcast<f32>(workgroup_state[block].aggregate);
    let inclusive_prefix = bitcast<f32>(workgroup_state[block].inclusive_prefix);

    if (state != 0) {
        switch (state) {
            case 1u: {
                if (aggregate != 0) {
                    return true;
                }
            }
            case 2u: {return true;}
            case 3u: {
                if (inclusive_prefix != 0) {
                    return true;
                }
            }
            case 4u: {return true;}
            default: {
                return false;
            }
        }
    }
    return false;
}

fn lookback(block: u32, tid: u32) {
    if (tid == 0) {
        prefix = 0.0;
    }

    if (block == 0) {
        workgroup_state[block].state = 4u;
        return;
    }

    var inspecting: u32 = block - 1;

    loop {

        var done: bool = false;
        var attempt: u32;
        for (attempt = 0u; attempt < 1024 && !done; attempt = attempt + 1) {
            let state = workgroup_state[inspecting].state;
            switch (state) {
                case 0u: {
                    reduce(inspecting, tid);
                }
                case 1u: {
                    let aggregate = bitcast<f32>(workgroup_state[inspecting].aggregate);
                    done = aggregate != 0.0;
                    if (tid == 0) {
                        prefix += aggregate;
                    }
                }
                case 2u: {
                }
                case 3u: {
                    let inclusive_prefix = bitcast<f32>(workgroup_state[inspecting].inclusive_prefix);
                    let aggregate = bitcast<f32>(workgroup_state[inspecting].aggregate);
                    if (tid == 0) {
                        prefix += inclusive_prefix + aggregate;
                    }
                    return;
                }
                case 4u: {
                    let aggregate = bitcast<f32>(workgroup_state[inspecting].aggregate);
                    if (tid == 0) {
                        prefix += aggregate;
                    }
                    return;
                }
                default: {
                    done = true;
                }
            }
        }
        if (inspecting == 0) {
            return;
        }
        inspecting = inspecting - 1;
        workgroupBarrier();
    }
}

fn scan(block: u32, tid: u32, prefix: f32) {
    let start = block * n;
    var offset: u32 = 1;

    // copy the data over
    shared_data[2*tid] = data[start + 2*tid];
    shared_data[2*tid + 1] = data[start + 2*tid + 1];

    for (var d: u32 = n >> 1; d > 0; d = d >> 1) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset*(2*tid + 1) - 1;
            let bi = offset*(2*tid + 2) - 1;
            shared_data[bi] = shared_data[bi] + shared_data[ai];
        }
        offset = offset * 2;
    }
    if (tid == 0) {
        shared_data[n - 1] = 0.0;
    }
    for (var d: u32 = 1; d < n; d = d * 2) {
        offset = offset >> 1;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset*(2*tid + 1) - 1;
            let bi = offset*(2*tid + 2) - 1;
            let temp = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] = shared_data[bi] + temp;
        }
    }
    workgroupBarrier();
    data[2*tid + start] = shared_data[2*tid] + prefix;
    data[2*tid + start + 1] = shared_data[2*tid + 1] + prefix;
}

fn get_block() -> u32 {
    return atomicAdd(&next_block, 1u);
}

var<workgroup> block: u32;
var<workgroup> prefix: f32;

const _a: u32 = 97u;
const _b: u32 = 98u;
const _c: u32 = 99u;
const _d: u32 = 100u;
const _e: u32 = 101u;
const _f: u32 = 102u;
const _g: u32 = 103u;
const _h: u32 = 104u;
const _i: u32 = 105u;
const _j: u32 = 106u;
const _k: u32 = 107u;
const _l: u32 = 108u;
const _m: u32 = 109u;
const _n: u32 = 110u;
const _o: u32 = 111u;
const _p: u32 = 112u;
const _q: u32 = 113u;
const _r: u32 = 114u;
const _s: u32 = 115u;
const _t: u32 = 116u;
const _u: u32 = 117u;
const _v: u32 = 118u;
const _w: u32 = 119u;
const _x: u32 = 120u;
const _y: u32 = 121u;
const _z: u32 = 122u;

@compute
@workgroup_size(2)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    if (local_id.x == 0) {
        block = get_block();
    }
    workgroupBarrier();

    reduce(block, local_id.x);

    lookback(block, local_id.x);

    if (local_id.x == 0) {
        if (prefix == 0.0) {
            workgroup_state[block].state = 4u;
        } else {
            workgroup_state[block].state = 3u;
            workgroup_state[block].inclusive_prefix = bitcast<u32>(prefix);
        }
    }

    scan(block, local_id.x, prefix);

    workgroupBarrier();
}

// only the 8 lowest bits are taken
fn debug_byte(block: u32, byte: u32) {
    let next_byte = workgroup_state[block].next_byte;
    let debug_byte_index = next_byte / 4;
    let debug_bit_index = next_byte % 4 * 8;

    let bitmask = (byte & 255u) << debug_bit_index;
    workgroup_state[block].debug_info[debug_byte_index] = workgroup_state[block].debug_info[debug_byte_index] | bitmask;
    workgroup_state[block].next_byte = workgroup_state[block].next_byte + 1;
}
