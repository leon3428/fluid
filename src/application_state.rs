use std::sync::Arc;

use pollster::FutureExt as _;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    fluid_simulation::FluidSimulation,
    pipelines::{
        create_field_render_pipeline, create_filed_compute_pipeline,
        create_particle_render_pipeline,
    },
};

// TODO: remove
const ROWS: u32 = 32;
const COLS: u32 = 32;
const TOP: f32 = -0.5;
const LEFT: f32 = -0.5;
const WINDOW_SIZE: u32 = 800;

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,

    simulation: FluidSimulation,
    particle_position_buffer: wgpu::Buffer,
    particle_density_buffer: wgpu::Buffer,
    field_texture: wgpu::Texture,
    field_compute_pipeline: wgpu::ComputePipeline,
    field_compute_bind_group: wgpu::BindGroup,

    field_render_pipeline: wgpu::RenderPipeline,
    field_render_bind_group: wgpu::BindGroup,

    window: Arc<Window>,
}

impl State {
    pub fn new(window: Window) -> Self {
        let window_arc = Arc::new(window);
        let size = window_arc.inner_size();
        let instance = Self::create_gpu_instance();
        let surface = instance.create_surface(window_arc.clone()).unwrap();
        let adapter = Self::create_adapter(instance, &surface);
        let (device, queue) = Self::create_device(&adapter);
        let surface_caps = surface.get_capabilities(&adapter);
        let config = Self::create_surface_config(size, surface_caps);
        let pipeline = create_particle_render_pipeline(&device, &config);
        surface.configure(&device, &config);

        let smoothing_radius = 0.04;
        let bound_damping = -0.5;
        let mass = 0.001;
        let viscosity = 0.001;

        let simulation = FluidSimulation::with_grid_initialization(
            smoothing_radius,
            bound_damping,
            mass,
            viscosity,
            ROWS,
            COLS,
            TOP,
            LEFT,
        );
        let (particle_position_buffer, particle_density_buffer) =
            Self::create_particle_buffers(&device, &simulation);
        let field_texture = Self::create_field_texture(&device, WINDOW_SIZE, WINDOW_SIZE);

        let field_texture_view = field_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let field_texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (field_compute_pipeline, field_compute_bind_group_layout) =
            create_filed_compute_pipeline(&device);
        let field_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Filed compute bind group"),
            layout: &field_compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&field_texture_view),
                },
            ],
        });

        let (field_render_pipeline, field_render_bind_group_layout) =
            create_field_render_pipeline(&device, &config);

        let field_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Filed render bind group"),
            layout: &field_render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&field_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&field_texture_sampler),
                },
            ],
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            simulation,
            particle_position_buffer,
            particle_density_buffer,
            field_texture,
            field_compute_pipeline,
            field_compute_bind_group,
            field_render_bind_group,
            field_render_pipeline,
            window: window_arc,
        }
    }

    fn create_particle_buffers(
        device: &wgpu::Device,
        simulation: &FluidSimulation,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle position buffer"),
            contents: simulation.positions_data(),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        let density_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle density buffer"),
            contents: simulation.density_data(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        (position_buffer, density_buffer)
    }

    fn create_field_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Field texture"),
            size: wgpu::Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_surface_config(
        size: PhysicalSize<u32>,
        capabilities: wgpu::SurfaceCapabilities,
    ) -> wgpu::SurfaceConfiguration {
        let surface_format = capabilities
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(capabilities.formats[0]);

        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        }
    }

    fn create_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .block_on()
            .unwrap()
    }

    fn create_adapter(instance: wgpu::Instance, surface: &wgpu::Surface) -> wgpu::Adapter {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .block_on()
            .unwrap()
    }

    fn create_gpu_instance() -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        })
    }

    pub fn update(&mut self, dt: f32) {
        self.simulation.update(dt);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Field compute pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.field_compute_pipeline);
            compute_pass.set_bind_group(0, &self.field_compute_bind_group, &[]);
            compute_pass.dispatch_workgroups((WINDOW_SIZE + 15) / 16, (WINDOW_SIZE + 15) / 16, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Field render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.field_render_pipeline);
            render_pass.set_bind_group(0, &self.field_render_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.particle_position_buffer.slice(..));
            render_pass.draw(0..4, 0..self.simulation.num_particles());
        }

        self.queue.write_buffer(
            &self.particle_position_buffer,
            0,
            self.simulation.positions_data(),
        );
        self.queue.write_buffer(
            &self.particle_density_buffer,
            0,
            self.simulation.density_data(),
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}
