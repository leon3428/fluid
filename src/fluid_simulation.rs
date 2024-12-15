use std::f32::consts::PI;

use nalgebra::{Normed, Vector2};

const DT: f32 = 0.001;
const REST_DENS: f32 = 1.0;
const GAS_CONST: f32 = 10.0;
pub const G: Vector2<f32> = Vector2::new(0.0, -1.0);

pub struct FluidSimulation {
    smoothing_radius: f32,
    bound_damping: f32,
    mass: f32,
    viscosity: f32,

    positions: Vec<Vector2<f32>>,
    velocities: Vec<Vector2<f32>>,
    densities: Vec<f32>,
    pressures: Vec<f32>,
}

impl FluidSimulation {
    pub fn with_grid_initialization(
        smoothing_radius: f32,
        bound_damping: f32,
        mass: f32,
        viscosity: f32,
        rows: u32,
        cols: u32,
        top: f32,
        left: f32,
    ) -> Self {
        let mut positions = Vec::with_capacity((rows * cols) as usize);
        let mut velocities = Vec::with_capacity((rows * cols) as usize);
        let mut densities = vec![0.0; (rows * cols) as usize];
        let mut pressures = vec![0.0; (rows * cols) as usize];

        for i in 0..rows {
            for j in 0..cols {
                let jitter_x = (rand::random::<f32>() - 0.5) / 50.0;
                let jitter_y = (rand::random::<f32>() - 0.5) / 50.0;

                positions.push(Vector2::new(
                    j as f32 * smoothing_radius * 0.95 + left + jitter_x,
                    i as f32 * smoothing_radius * 0.95 + top + jitter_y,
                ));

                velocities.push(Vector2::new(0.0, 0.0));
            }
        }

        Self {
            smoothing_radius,
            bound_damping,
            mass,
            viscosity,
            positions,
            velocities,
            densities,
            pressures,
        }
    }

    pub fn positions_data(&self) -> &[u8] {
        let len = self.positions.len() * std::mem::size_of::<Vector2<f32>>();
        let ptr = self.positions.as_ptr() as *const u8;

        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn num_particles(&self) -> u32 {
        self.positions.len() as u32
    }

    pub fn update(&mut self, dt: f32) {
        self.compute_density();
        // println!("{:?}", self.pressures);

        let spiky_grad: f32 = -10.0 / (PI * self.smoothing_radius.powi(5));
        let visc_lap: f32 = 40.0 / (PI * self.smoothing_radius.powi(5));

        for i in 0..self.densities.len() {
            let mut force = Vector2::new(0.0, 0.0);
            for j in 0..self.densities.len() {
                if i == j {
                    continue;
                }

                let r = self.positions[j] - self.positions[i];
                let r_norm = r.norm();
                if r_norm < self.smoothing_radius {
                    force += r.normalize() * self.mass * (self.pressures[i] + self.pressures[j])
                        / (2.0 * self.densities[j] + 1e-6)
                        * spiky_grad
                        * (self.smoothing_radius - r_norm).powi(2);
                    force += self.viscosity * self.mass * (self.velocities[j] - self.velocities[i]) / ( self.densities[j] + 1e-6) * visc_lap * (self.smoothing_radius - r_norm);
                }
            }
            force += G * self.densities[i];

            self.velocities[i] += force * (DT / (self.densities[i] + 1e-6) );
            self.positions[i] += self.velocities[i] * DT;

            if self.positions[i].x - self.smoothing_radius < -1.0 {
                self.velocities[i].x *= self.bound_damping;
                self.positions[i].x = -1.0 + self.smoothing_radius;
            }

            if self.positions[i].x + self.smoothing_radius > 1.0 {
                self.velocities[i].x *= self.bound_damping;
                self.positions[i].x = 1.0 - self.smoothing_radius;
            }

            if self.positions[i].y - self.smoothing_radius < -1.0 {
                self.velocities[i].y *= self.bound_damping;
                self.positions[i].y = -1.0 + self.smoothing_radius;
            }

            if self.positions[i].y + self.smoothing_radius > 1.0 {
                self.velocities[i].y *= self.bound_damping;
                self.positions[i].y = 1.0 - self.smoothing_radius;
            }
        }
    }

    fn compute_density(&mut self) {
        let smoothing_radius_sq = self.smoothing_radius * self.smoothing_radius;
        let poly6: f32 = 4.0 / (PI * self.smoothing_radius.powi(8));

        for i in 0..self.densities.len() {
            self.densities[i] = 0.0;

            for j in 0..self.densities.len() {
                let r = self.positions[j] - self.positions[i];
                let r_sq = r.norm_squared();

                if r_sq < smoothing_radius_sq {
                    self.densities[i] += self.mass * poly6 * (smoothing_radius_sq - r_sq).powi(3);
                }
            }

            self.pressures[i] = GAS_CONST * (self.densities[i] - REST_DENS);
        }
    }
}
