mod cpu;

use wgpu::{
    util::DeviceExt, BufferUsages, CommandEncoderDescriptor, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, PipelineCompilationOptions, RequestAdapterOptions,
};

use std::{mem::size_of, num::NonZeroU64, time::Instant};

fn main() {
    cpu::run();
    pollster::block_on(run());
}

const SIZE: usize = 256 * 256;

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
    state.scan(&buffer, SIZE as u32).await;
    println!("gpu compute took {:?}", start.elapsed());

    let data = load_buffer(
        &buffer,
        &state.debug_buffer,
        SIZE,
        &state.device,
        &state.queue,
    )
    .await;
    println!("\t={}", data.last().unwrap());
}

struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    debug_buffer: wgpu::Buffer,
    scan_compute_pipeline: wgpu::ComputePipeline,
    add_compute_pipeline: wgpu::ComputePipeline,
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
                    // sums buffer
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
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
        let add_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "scan_main",
                compilation_options: PipelineCompilationOptions::default(),
            });
        let scan_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "add_main",
                compilation_options: PipelineCompilationOptions::default(),
            });

        let debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Debug Buffer"),
            contents: bytemuck::cast_slice(&[0f32; SIZE]),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        Self {
            device,
            queue,
            bind_group_layout,
            debug_buffer,
            add_compute_pipeline,
            scan_compute_pipeline,
        }
    }

    async fn scan(&self, to_scan: &wgpu::Buffer, size: u32) {
        let mut commands = Vec::new();
        self.scan_into(to_scan, size, &mut commands).await;
        self.queue.submit(commands);
    }

    #[async_recursion::async_recursion]
    async fn scan_into(
        &self,
        to_scan: &wgpu::Buffer,
        size: u32,
        to_execute: &mut Vec<wgpu::CommandBuffer>,
    ) {
        let elements_per_workgroup = 64 * 2;
        let workgroups = ((size as f32 / elements_per_workgroup as f32).ceil() as u32).max(1);
        let sums = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sums Buffer"),
                contents: &vec![0u8; workgroups as usize * size_of::<f32>()],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
                    resource: sums.as_entire_binding(),
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
            pass.set_pipeline(&self.add_compute_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        to_execute.push(encoder.finish());

        if workgroups > 1 {
            self.scan_into(&sums, workgroups, to_execute).await;

            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_pipeline(&self.scan_compute_pipeline);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            to_execute.push(encoder.finish());
        }
    }

    fn ones_buffer(&self, size: usize) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vec![1f32; size]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }
}

async fn clear_buffer(buffer: &wgpu::Buffer, length: usize, queue: &wgpu::Queue) {
    queue.write_buffer(buffer, 0, &vec![0u8; length * 4]);
    queue.submit([]);
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
