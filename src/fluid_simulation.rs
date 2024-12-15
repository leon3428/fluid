#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    position: [f32; 2],
}

pub struct FluidSimulation {
    particles: Vec<Particle>,
}

impl FluidSimulation {
    pub fn with_grid_initialization(rows: u32, cols: u32, top: f32, left: f32, offset: f32) -> Self {
        let mut particles = Vec::with_capacity((rows * cols) as usize);

        for i in 0..rows {
            for j in 0..cols {
                particles.push(Particle {
                    position: [j as f32 * offset + left, i as f32 * offset + top],
                });
            }
        }

        Self { particles }
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn num_particles(&self) -> u32 {
        self.particles.len() as u32
    }
}
