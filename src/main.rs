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

const SIZE: usize = 256 * 256 * 4;

async fn run() {
    // Set up surface
    let instance = Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .unwrap();
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

    let buffer1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compute Buffer 1"),
        contents: bytemuck::cast_slice(&[1f32; SIZE]),
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let buffer2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compute Buffer 2"),
        contents: bytemuck::cast_slice(&[0f32; SIZE]),
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Output Compute Buffer"),
        contents: bytemuck::cast_slice(&[0f32; SIZE]),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    let stride_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Stride Buffer"),
        contents: bytemuck::cast_slice(&[2u32]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("./shader.wgsl"));
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: PipelineCompilationOptions::default(),
    });

    let bind_group1to2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer2.as_entire_binding(),
            },
        ],
    });

    let bind_group2to1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer2.as_entire_binding(),
            },
        ],
    });

    let start = Instant::now();

    let mut elements = SIZE;

    let mut buffer1input = true;

    let elements_per_workgroup = 256 * 2;

    let mut commands = Vec::new();

    while elements / 2 > 0 {
        let workgroups = (elements as u32 / elements_per_workgroup).max(1);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            let bind_group = if buffer1input {
                &bind_group1to2
            } else {
                &bind_group2to1
            };
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&compute_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        commands.push(encoder.finish());
        elements /= elements_per_workgroup as usize;
        buffer1input = !buffer1input;
    }
    queue.submit(commands);
    println!("gpu sum took {:?}", start.elapsed());
    let data = load_buffer(
        if buffer1input { &buffer1 } else { &buffer2 },
        &output_buffer,
        1,
        &device,
        &queue,
    )
    .await
    .first()
    .copied()
    .unwrap();
    println!("\t= {data}");
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
