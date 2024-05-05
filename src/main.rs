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

const SIZE: usize = 1024 * 256;

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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                },
                count: None,
            },
        ],
    });

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Compute Buffer"),
        contents: bytemuck::cast_slice(&[1f32; SIZE]),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: stride_buffer.as_entire_binding(),
            },
        ],
    });
    let mut stride = 2;

    let start = Instant::now();

    while stride <= SIZE {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&compute_pipeline);
            pass.dispatch_workgroups(((SIZE / stride / 256) as u32).max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &input_buffer,
            0,
            &output_buffer,
            0,
            (SIZE * size_of::<f32>()) as wgpu::BufferAddress,
        );
        queue.write_buffer(&stride_buffer, 0, bytemuck::cast_slice(&[stride as u32]));
        queue.submit(Some(encoder.finish()));

        stride *= 2;
    }
    let buffer_slice = output_buffer.slice(..);

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

    output_buffer.unmap();

    println!("total sum took {:?}", start.elapsed());
    println!("the total value is {:?}", data.last().unwrap());

    //    for number in data {
    //        print!("{} ", number as u32);
    //    }
    //    println!();
}
