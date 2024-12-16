@group(0) @binding(0) var<storage, read> particle_positions: array<vec2<f32>>; 
@group(0) @binding(1) var<storage, read> particle_densities: array<f32>; 
@group(0) @binding(2) var density_field: texture_storage_2d<r32float, write>; // Output texture data

const PI: f32 = 3.1415927;
const H: f32 = 0.04;
const HSQ: f32 = H * H;
const POLY6: f32 = 4.0 / (PI * H * H * H * H * H * H * H * H);
const MASS: f32 = 0.001;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resolution = 800u; // Resolution of the density field
    let grid_size = f32(resolution);

    if (global_id.x >= resolution || global_id.y >= resolution) {
        return;
    }

    // Compute normalized coordinates for the grid cell
    let grid_pos = vec2<f32>(
        f32(global_id.x) / grid_size * 2.0 - 1.0,
        f32(global_id.y) / grid_size * 2.0 - 1.0,
    );

    var density: f32 = 0.0;

    // Sum contributions from all particles
    for (var i = 0u; i < arrayLength(&particle_positions); i++) {
        let dist = length(grid_pos - particle_positions[i]);
        let dist_sq = dist * dist;

        if (dist < H) {
            // Example: Poly6 kernel contribution
            let contribution = MASS * POLY6 * (HSQ - dist_sq) * (HSQ - dist_sq) * (HSQ - dist_sq);
            density += contribution;
        }
    }

    // Write the computed density to the output texture
    textureStore(density_field, global_id.xy, vec4f(density, 0.0, 0.0, 1.0));
}
