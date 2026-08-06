// Particles, trails & sheets: the persistent shared buffer, drawn as one
// opaque-pass item (see particles_and_trails.rs). Vertices arrive in world
// space with atlas UVs and tint baked CPU-side; this shader projects, samples
// and hard-masks at 0.5 — the unlit Mask(0.5) StandardMaterial it replaced.
//
// Tint alpha doubles as a lit flag: alpha > 1.0 (sheets) shades with one sun
// + flat ambient off the smooth vertex normal, flipped toward the camera
// because cloth is double-sided. Particles and trails keep alpha <= 1.0 and
// stay fullbright (their normals are zero).

#import bevy_render::view::View

struct FxLight {
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var atlas_texture: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(0) @binding(3) var<uniform> light: FxLight;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) normal: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) normal: vec3<f32>,
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_position = view.clip_from_world * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.world_position = in.position;
    out.normal = in.normal;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let lit = in.color.a > 1.5;
    let color = textureSample(atlas_texture, atlas_sampler, in.uv)
        * vec4<f32>(in.color.rgb, min(in.color.a, 1.0));
#ifndef ALPHA_TO_COVERAGE
    if color.a < 0.5 {
        discard;
    }
#endif
    var rgb = color.rgb;
    if lit {
        var normal = normalize(in.normal);
        if dot(normal, view.world_position - in.world_position) < 0.0 {
            normal = -normal;
        }
        rgb *= light.ambient.rgb
            + light.sun_color.rgb * max(dot(normal, light.sun_dir.xyz), 0.0);
    }
#ifdef ALPHA_TO_COVERAGE
    // MSAA turns this alpha into the coverage mask (see the specializer).
    return vec4<f32>(rgb, color.a);
#else
    return vec4<f32>(rgb, 1.0);
#endif
}
