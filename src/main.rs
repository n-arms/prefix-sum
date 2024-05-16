mod cpu;

use wgpu::{
    util::DeviceExt, BufferUsages, CommandEncoderDescriptor, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, PipelineCompilationOptions, RequestAdapterOptions,
};

use core::fmt;
use std::{iter, mem::size_of, num::NonZeroU64, os::unix::net::UnixDatagram, time::Instant};

fn main() {
    cpu::run();
    pollster::block_on(run());
}

const SIZE: usize = 16;

async fn run() {
    // Set up surface
    let instance = Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .unwrap();
    let state = State::new(&adapter).await;
    let buffer = state.ones_buffer(SIZE);
    let start = Instant::now();
    let (workgroup_state, next_block) = state.scan(&buffer, &state.device, SIZE as u32).await;
    println!("gpu compute took {:?}", start.elapsed());

    let prefix_scan = load_buffer(
        &buffer,
        &state.debug_buffer,
        SIZE,
        &state.device,
        &state.queue,
    )
    .await;
    //print_data(data);
    //println!("\t={}", data.last().unwrap());

    let states = load_workgroup_state_buffer(
        &workgroup_state,
        &state.debug_buffer,
        SIZE / 4,
        &state.device,
        &state.queue,
    )
    .await;
    //print_states(data);

    for (scans, state) in prefix_scan.chunks_exact(4).zip(states) {
        for scan in scans {
            print!("{} ", scan);
        }
        println!("{:?}", state);
    }

    let data = load_u32_buffer(
        &next_block,
        &state.debug_buffer,
        1,
        &state.device,
        &state.queue,
    )
    .await;
    println!("next block: {}", data.first().unwrap());
}

struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    debug_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WorkGroupState {
    // 0: no work has been done, 1: the aggregate field contains the aggregate, 2: the aggregate is 0, 3: the inclusive prefix field contains the inclusive prefix, 4: the inclusive prefix is 0
    state: u32,
    aggregate: f32,
    inclusive_prefix: f32,
    debug_info: [u8; 1024],
    next_byte: u32,
}

impl State {
    async fn new(adapter: &wgpu::Adapter) -> Self {
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // data buffer
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // workgroup progress buffer
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(size_of::<WorkGroupState>() as wgpu::BufferAddress)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // workgroup progress buffer
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(size_of::<u32>() as wgpu::BufferAddress).unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("./shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: PipelineCompilationOptions::default(),
        });

        let debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Debug Buffer"),
            contents: bytemuck::cast_slice(&[0f32; 1024 * 1024]),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        Self {
            device,
            queue,
            bind_group_layout,
            debug_buffer,
            pipeline,
        }
    }

    async fn scan(
        &self,
        to_scan: &wgpu::Buffer,
        device: &wgpu::Device,
        size: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let mut commands = Vec::new();
        let result = self.scan_into(to_scan, device, size, &mut commands).await;
        self.queue.submit(commands);
        return result;
    }

    async fn scan_into(
        &self,
        to_scan: &wgpu::Buffer,
        device: &wgpu::Device,
        size: u32,
        to_execute: &mut Vec<wgpu::CommandBuffer>,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let elements_per_workgroup = 2 * 2;
        let workgroups = ((size as f32 / elements_per_workgroup as f32).ceil() as u32).max(1);
        let workgroup_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Workgroup State Buffer"),
            size: (workgroups as usize * size_of::<WorkGroupState>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let next_block_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Block Buffer"),
            size: size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: to_scan.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: workgroup_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: next_block_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        to_execute.push(encoder.finish());

        return (workgroup_state_buffer, next_block_buffer);
    }

    fn ones_buffer(&self, size: usize) -> wgpu::Buffer {
        let data: Vec<_> = (1..size + 1).map(|x| x as f32).collect();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }
}

async fn clear_buffer(buffer: &wgpu::Buffer, length: usize, queue: &wgpu::Queue) {
    queue.write_buffer(buffer, 0, &vec![0u8; length * 4]);
    queue.submit([]);
}
async fn load_u32_buffer(
    source: &wgpu::Buffer,
    temp: &wgpu::Buffer,
    elements: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<u32> {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(
        source,
        0,
        temp,
        0,
        (elements * size_of::<u32>()) as wgpu::BufferAddress,
    );
    queue.submit(Some(encoder.finish()));
    let buffer_slice = temp.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice
        .get_mapped_range()
        .as_ref()
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
        .take(elements)
        .collect::<Vec<_>>();

    temp.unmap();

    data
}
async fn load_buffer(
    source: &wgpu::Buffer,
    temp: &wgpu::Buffer,
    elements: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<f32> {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(
        source,
        0,
        temp,
        0,
        (elements * size_of::<f32>()) as wgpu::BufferAddress,
    );
    queue.submit(Some(encoder.finish()));
    let buffer_slice = temp.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice
        .get_mapped_range()
        .as_ref()
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .take(elements)
        .collect::<Vec<_>>();

    temp.unmap();

    data
}

fn print_data(data: Vec<f32>) {
    for elem in data {
        print!("{} ", elem);
    }
    println!();
}

async fn load_workgroup_state_buffer(
    source: &wgpu::Buffer,
    temp: &wgpu::Buffer,
    elements: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<WorkGroupState> {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(
        source,
        0,
        temp,
        0,
        (elements * size_of::<WorkGroupState>()) as wgpu::BufferAddress,
    );
    queue.submit(Some(encoder.finish()));
    let buffer_slice = temp.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice
        .get_mapped_range()
        .as_ref()
        .chunks_exact(size_of::<WorkGroupState>())
        .map(|bytes| {
            let state = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
            let aggregate = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
            let inclusive_prefix = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());
            let debug_info = bytes[12..]
                .chunks_exact(4)
                .flat_map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()).to_le_bytes())
                .chain(iter::repeat(0))
                .take(1024)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            WorkGroupState {
                state,
                aggregate,
                inclusive_prefix,
                debug_info,
                next_byte: 0,
            }
        })
        .take(elements)
        .collect();

    temp.unmap();

    data
}

fn print_states(data: Vec<WorkGroupState>) {
    for elem in data {
        print!("{:?} ", elem);
    }
    println!();
}

impl WorkGroupState {
    pub fn state_name(&self) -> &'static str {
        match self.state {
            0 => "Uninit",
            1 => "Aggregated",
            2 => "Zero Aggregated",
            3 => "Prefixed",
            4 => "Zero Prefixed",
            _ => "Unknown",
        }
    }
}

impl fmt::Debug for WorkGroupState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let debug: Vec<_> = self
            .debug_info
            .iter()
            .copied()
            .take_while(|byte| *byte != b'\0')
            .collect();
        let debug_str = String::from_utf8(debug).unwrap();
        f.debug_struct("WorkGroupState")
            .field("state", &self.state_name())
            .field("aggregate", &self.aggregate)
            .field("inclusive_prefix", &self.inclusive_prefix)
            .field("debug info", &debug_str)
            .finish()
    }
}
