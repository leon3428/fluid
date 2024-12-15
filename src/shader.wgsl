struct VertexInput {
    @location(0) particle_pos: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) center: vec2<f32>,
};

const SIZE: f32 = 0.01;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    vertex_input: VertexInput
) -> VertexOutput {
    var out: VertexOutput;

    var quad_vertices: array<vec2<f32>, 4> = array(
        vec2f(-SIZE, -SIZE),
        vec2f( SIZE, -SIZE),
        vec2f(-SIZE,  SIZE),
        vec2f( SIZE,  SIZE),
    );

    let pos = quad_vertices[in_vertex_index] + vertex_input.particle_pos;

    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.center = quad_vertices[in_vertex_index];
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if length(in.center) < SIZE {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    } else {
        discard;
    }
}