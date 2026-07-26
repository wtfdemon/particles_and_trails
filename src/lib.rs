//! Every billboard particle and ribbon trail in one fixed-size buffer: one
//! atlas texture, one custom opaque-phase draw call.
//!
//! The CPU rewrites the whole vertex range every frame into a preallocated
//! scratch (the `bevy_trail` trick — a bounded memcpy beats growing meshes
//! and per-entity extraction). Because everything is re-sent anyway,
//! animation frames, movement, spawning and despawning are plain CPU list
//! operations: a particle exists only as an entry in the [`Particles`]
//! resource, a trail as a [`Trail`] component sampled from its
//! `GlobalTransform`.
//!
//! The GPU side is a custom [`Opaque3d`] phase item (bevy's
//! `custom_phase_item` example pattern): two `RawBufferVec`s allocated once at
//! capacity and refilled with `write_buffer` — never recreated. This used to
//! be a `Mesh` asset dirtied per frame, which made bevy create a fresh staging
//! buffer every frame; Firefox's WebGPU process fell behind processing the
//! churn and ran out of memory (see webgpu_buffer_churn_test.html history).
//!
//! Alpha is a hard shader `discard` under 0.5 with depth writes, in the opaque
//! pass: unsorted, so buffer order never matters. On overflow trails win and
//! the oldest particles drop — the Quake particle budget.

use std::collections::VecDeque;

use bevy::{
    asset::{embedded_asset, AssetId},
    core_pipeline::core_3d::{
        Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, CORE_3D_DEPTH_FORMAT,
    },
    ecs::system::{
        lifetimeless::{Read, SRes},
        SystemParamItem,
    },
    image::{ImageLoaderSettings, ImageSampler},
    mesh::VertexBufferLayout,
    prelude::*,
    render::{
        mesh::allocator::MeshSlabs,
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
            RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases,
        },
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingType,
            BufferBindingType, BufferId, BufferUsages, Canonical, ColorTargetState, ColorWrites,
            CompareFunction, DepthStencilState, FragmentState, IndexFormat, PipelineCache,
            RawBufferVec, RenderPipeline, RenderPipelineDescriptor, ShaderStages, ShaderType,
            Specializer, SpecializerKey, TextureFormat, TextureSampleType, TextureViewDimension,
            Variants, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
        texture::GpuImage,
        view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    },
    transform::TransformSystems,
};
use bytemuck::{Pod, Zeroable};

const SHADER_PATH: &str = "embedded://bevy_particles_and_trails/fx_buffer.wgsl";

/// A rectangle of the atlas holding `frames` equally sized cells, read
/// left-to-right then top-to-bottom.
#[derive(Clone, Copy)]
pub struct SpriteAnim {
    /// Atlas region in UVs (0..1 over the whole atlas).
    pub rect: Rect,
    pub frames: u32,
    /// Cells per row; 0 means a single row.
    pub columns: u32,
    /// Playback speed; 0 locks frame 0.
    pub fps: f32,
}

impl SpriteAnim {
    /// The whole atlas as one static frame — the default sprite.
    pub const FULL: Self = Self::single(Rect {
        min: Vec2::ZERO,
        max: Vec2::ONE,
    });

    pub const fn single(rect: Rect) -> Self {
        Self {
            rect,
            frames: 1,
            columns: 0,
            fps: 0.0,
        }
    }

    /// Length of one playthrough; infinite for static sprites.
    pub fn duration(&self) -> f32 {
        if self.fps > 0.0 {
            self.frames as f32 / self.fps
        } else {
            f32::INFINITY
        }
    }

    /// UV rect of the frame playing at `age` seconds (looping).
    fn frame_rect(&self, age: f32) -> Rect {
        let frames = self.frames.max(1);
        let columns = if self.columns == 0 { frames } else { self.columns };
        let frame = if self.fps > 0.0 {
            (age * self.fps) as u32 % frames
        } else {
            0
        };
        let cell = self.rect.size() / Vec2::new(columns as f32, frames.div_ceil(columns) as f32);
        let min = self.rect.min + cell * Vec2::new((frame % columns) as f32, (frame / columns) as f32);
        Rect::from_corners(min, min + cell)
    }
}

#[derive(Clone)]
pub struct Particle {
    pub sprite: SpriteAnim,
    pub position: Vec3,
    pub velocity: Vec3,
    /// Downward acceleration in units/s².
    pub gravity: f32,
    /// World-space quad size.
    pub size: Vec2,
    /// Roll around the view axis, radians.
    pub rotation: f32,
    /// Multiplies the texture color.
    pub tint: LinearRgba,
    /// Seconds until removal; `INFINITY` loops until despawned by id.
    pub lifetime: f32,
    /// Size multiplier reached at end of life (1.0 = constant size).
    pub end_scale: f32,
    /// AE-style position wiggle: max offset in world units, 0 = off.
    pub wiggle_amp: f32,
    /// Wiggle frequency in Hz.
    pub wiggle_freq: f32,
    // Managed by `update_particles`; construct with `..default()`.
    pub age: f32,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            sprite: SpriteAnim::FULL,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            gravity: 0.0,
            size: Vec2::ONE,
            rotation: 0.0,
            tint: LinearRgba::WHITE,
            lifetime: f32::INFINITY,
            end_scale: 1.0,
            wiggle_amp: 0.0,
            wiggle_freq: 1.0,
            age: 0.0,
        }
    }
}

impl Particle {
    /// Plays an animated sprite once, then despawns itself.
    pub fn one_shot(sprite: SpriteAnim, position: Vec3, size: Vec2) -> Self {
        Self {
            sprite,
            position,
            size,
            lifetime: sprite.duration(),
            ..default()
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ParticleId(u64);

/// All live particles. Spawning is pushing an entry; no entities involved.
#[derive(Resource, Default)]
pub struct Particles {
    live: Vec<(ParticleId, Particle)>,
    next: u64,
}

impl Particles {
    pub fn spawn(&mut self, particle: Particle) -> ParticleId {
        self.next += 1;
        let id = ParticleId(self.next);
        self.live.push((id, particle));
        id
    }

    /// Remove a particle early — how looping particles end.
    #[allow(dead_code)]
    pub fn despawn(&mut self, id: ParticleId) {
        self.live.retain(|(i, _)| *i != id);
    }

    /// Mutate a live particle, e.g. to move a looping emitter.
    #[allow(dead_code)]
    pub fn get_mut(&mut self, id: ParticleId) -> Option<&mut Particle> {
        self.live.iter_mut().find(|(i, _)| *i == id).map(|(_, p)| p)
    }
}

struct TrailPoint {
    position: Vec3,
    timestamp: f32,
}

/// Ribbon trail emitter; attach to any entity with a `GlobalTransform`.
/// Width tapers to zero at the tail. The sprite may be animated.
#[derive(Component)]
pub struct Trail {
    pub sprite: SpriteAnim,
    pub tint: LinearRgba,
    pub max_points: usize,
    pub width: f32,
    /// Seconds a point lives before the tail eats it.
    pub max_age: f32,
    timer: Timer,
    points: VecDeque<TrailPoint>,
}

impl Trail {
    pub fn new(sprite: SpriteAnim, max_points: usize, emit_rate: f32, width: f32) -> Self {
        Self {
            sprite,
            tint: LinearRgba::WHITE,
            max_points,
            width,
            max_age: 5.0,
            timer: Timer::from_seconds(1.0 / emit_rate, TimerMode::Repeating),
            points: VecDeque::new(),
        }
    }

    pub fn with_max_age(mut self, max_age: f32) -> Self {
        self.max_age = max_age;
        self
    }
}

pub struct ParticlesAndTrailsPlugin {
    /// Vertex capacity of the shared buffer: 4 per particle, 2 per trail point.
    pub capacity: usize,
    /// Asset path of the atlas every sprite is a region of. Loaded with a
    /// nearest sampler — this is a retro-sprite renderer.
    pub atlas: String,
}

impl Plugin for ParticlesAndTrailsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "fx_buffer.wgsl");
        app.init_resource::<Particles>()
            .insert_resource(FxAtlasPath(self.atlas.clone()))
            .insert_resource(FxScratch::with_capacity(self.capacity))
            .add_systems(Startup, setup)
            .add_systems(
                PostUpdate,
                (update_particles, update_trails, rebuild)
                    .chain()
                    .after(TransformSystems::Propagate),
            );

        // Absent in tests and on the server.
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_render_command::<Opaque3d, DrawFxCommands>()
            .add_systems(RenderStartup, init_fx_pipeline)
            .add_systems(ExtractSchedule, extract_fx)
            .add_systems(
                Render,
                (
                    prepare_fx_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_fx_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    queue_fx.in_set(RenderSystems::Queue),
                ),
            );
    }
}

/// One vertex of the shared buffer; layout mirrored by `fx_buffer.wgsl` and
/// the pipeline's `VertexBufferLayout`.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FxVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

/// CPU-built geometry for this frame, cloned into the render world whole.
/// Only ever holds up to `vertex_capacity` vertices.
#[derive(Resource, Clone)]
struct FxScratch {
    vertices: Vec<FxVertex>,
    indices: Vec<u32>,
    vertex_capacity: usize,
}

impl FxScratch {
    fn with_capacity(vertex_capacity: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(vertex_capacity),
            indices: Vec::with_capacity(vertex_capacity * 3 / 2),
            vertex_capacity,
        }
    }
}

/// The atlas image handle, also cloned into the render world for bind-group
/// creation.
#[derive(Resource, Clone)]
struct FxAtlas(Handle<Image>);

#[derive(Resource)]
struct FxAtlasPath(String);

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, path: Res<FxAtlasPath>) {
    let atlas = asset_server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.sampler = ImageSampler::nearest();
        })
        .load(path.0.clone());
    commands.insert_resource(FxAtlas(atlas));
}

fn update_particles(time: Res<Time>, mut particles: ResMut<Particles>) {
    let dt = time.delta_secs();
    particles.live.retain_mut(|(_, p)| {
        p.age += dt;
        if p.age >= p.lifetime {
            return false;
        }
        p.velocity.y -= p.gravity * dt;
        let velocity = p.velocity;
        p.position += velocity * dt;
        let scale = 1.0
            + (p.end_scale - 1.0)
                * if p.lifetime.is_finite() {
                    (p.age / p.lifetime).clamp(0.0, 1.0)
                } else {
                    0.0
                };
        p.position.y = p.position.y.max(p.size.y * scale * 0.5);
        true
    });
}

fn update_trails(time: Res<Time>, mut trails: Query<(&mut Trail, &GlobalTransform)>) {
    let now = time.elapsed_secs();
    for (mut trail, transform) in &mut trails {
        let position = transform.translation();
        trail.timer.tick(time.delta());
        if trail.timer.just_finished() || trail.points.is_empty() {
            trail.points.push_back(TrailPoint {
                position,
                timestamp: now,
            });
            while trail.points.len() > trail.max_points {
                trail.points.pop_front();
            }
        }
        // The head hugs the emitter between emissions.
        if let Some(head) = trail.points.back_mut() {
            head.position = position;
        }
        while trail
            .points
            .front()
            .is_some_and(|p| now - p.timestamp > trail.max_age)
        {
            trail.points.pop_front();
        }
    }
}

fn rebuild(
    scratch: ResMut<FxScratch>,
    particles: Res<Particles>,
    trails: Query<&Trail>,
    camera: Query<&GlobalTransform, With<Camera3d>>,
    time: Res<Time>,
) {
    let scratch = scratch.into_inner();
    scratch.vertices.clear();
    scratch.indices.clear();

    if let Some(camera) = camera.iter().next() {
        let camera_position = camera.translation();
        let now = time.elapsed_secs();
        // Trails first: few and persistent. Particles fill what's left,
        // newest first, so bursts evict the oldest when the buffer is full.
        for trail in &trails {
            emit_trail(scratch, trail, camera_position, now);
        }
        let camera_rotation = camera.rotation();
        for (id, particle) in particles.live.iter().rev() {
            if scratch.vertices.len() + 4 > scratch.vertex_capacity {
                break;
            }
            emit_particle(scratch, particle, camera_rotation, id.0 as f32 * 39.77);
        }
    }
}

fn hash(n: i32) -> f32 {
    ((n as f32 * 127.1).sin() * 43758.547).fract()
}

/// Smooth 1D value noise in [-1, 1], one octave.
fn noise1(t: f32) -> f32 {
    let i = t.floor();
    let f = t - i;
    let s = f * f * (3.0 - 2.0 * f);
    let (a, b) = (hash(i as i32), hash(i as i32 + 1));
    (a + (b - a) * s) * 2.0 - 1.0
}

fn emit_particle(scratch: &mut FxScratch, particle: &Particle, camera_rotation: Quat, seed: f32) {
    let mut center = particle.position;
    if particle.wiggle_amp > 0.0 {
        // AE-style wiggle: two octaves of smooth 1D value noise per axis,
        // offset in the noise domain so the axes don't correlate.
        let t = particle.age * particle.wiggle_freq + seed;
        let w = |axis: f32| noise1(t + axis * 100.0) + 0.5 * noise1(t * 2.0 + axis * 200.0);
        center += Vec3::new(w(1.0), w(2.0), w(3.0)) * (particle.wiggle_amp / 1.5);
    }

    let right = camera_rotation * Vec3::X;
    let up = camera_rotation * Vec3::Y;
    let (sin, cos) = particle.rotation.sin_cos();
    let (bx, by) = (right * cos + up * sin, up * cos - right * sin);

    let life = if particle.lifetime.is_finite() && particle.lifetime > 0.0 {
        particle.age / particle.lifetime
    } else {
        0.0
    };
    let half = particle.size * 0.5 * (1.0 + (particle.end_scale - 1.0) * life);
    let uv = particle.sprite.frame_rect(particle.age);
    let tint = particle.tint.to_f32_array();

    let base = scratch.vertices.len() as u32;
    for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let position = center + bx * (half.x * sx) + by * (half.y * sy);
        scratch.vertices.push(FxVertex {
            position: position.to_array(),
            // +Y on the quad is the top of the sprite, which is v = min.
            uv: [
                if sx > 0.0 { uv.max.x } else { uv.min.x },
                if sy > 0.0 { uv.min.y } else { uv.max.y },
            ],
            color: tint,
        });
    }
    scratch
        .indices
        .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn emit_trail(scratch: &mut FxScratch, trail: &Trail, camera_position: Vec3, now: f32) {
    let n = trail.points.len();
    if n < 2 || scratch.vertices.len() + n * 2 > scratch.vertex_capacity {
        return;
    }
    let uv = trail.sprite.frame_rect(now);
    let tint = trail.tint.to_f32_array();
    let half_width = trail.width * 0.5;

    let base = scratch.vertices.len() as u32;
    for (i, point) in trail.points.iter().enumerate() {
        let progress = i as f32 / (n - 1) as f32;
        let right = right_vector(&trail.points, i, camera_position);
        let width = half_width * progress; // tapers to nothing at the tail
        for (side, v) in [(-1.0, uv.min.y), (1.0, uv.max.y)] {
            scratch.vertices.push(FxVertex {
                position: (point.position + right * (width * side)).to_array(),
                uv: [uv.min.x + (uv.max.x - uv.min.x) * progress, v],
                color: tint,
            });
        }
    }
    for i in 0..(n - 1) as u32 {
        let q = base + i * 2;
        scratch.indices.extend([q, q + 1, q + 2, q + 1, q + 3, q + 2]);
    }
}

fn right_vector(points: &VecDeque<TrailPoint>, i: usize, camera_position: Vec3) -> Vec3 {
    let dir = trail_direction(points, i);
    let to_camera = (camera_position - points[i].position).normalize_or_zero();
    let right = dir.cross(to_camera).normalize_or_zero();
    if right != Vec3::ZERO {
        return right;
    }
    let fallback = if dir.dot(Vec3::Y).abs() < 0.9 {
        dir.cross(Vec3::Y)
    } else {
        dir.cross(Vec3::X)
    }
    .normalize_or_zero();
    if fallback != Vec3::ZERO { fallback } else { Vec3::X }
}

fn trail_direction(points: &VecDeque<TrailPoint>, i: usize) -> Vec3 {
    if i == 0 {
        (points[1].position - points[0].position).normalize_or_zero()
    } else if i == points.len() - 1 {
        (points[i].position - points[i - 1].position).normalize_or_zero()
    } else {
        ((points[i].position - points[i - 1].position)
            + (points[i + 1].position - points[i].position))
            .normalize_or_zero()
    }
}

// ============================================================================
// Render world: persistent buffers, one bind group, one opaque phase item.
// Nothing here creates GPU resources per frame — that churn is what drowned
// Firefox's WebGPU process.
// ============================================================================

/// Copies this frame's scratch + the atlas handle into the render world.
fn extract_fx(
    mut commands: Commands,
    scratch: Extract<Option<Res<FxScratch>>>,
    atlas: Extract<Option<Res<FxAtlas>>>,
) {
    if let Some(scratch) = scratch.as_ref() {
        commands.insert_resource(FxScratch::clone(scratch));
    }
    if let Some(atlas) = atlas.as_ref() {
        commands.insert_resource(FxAtlas::clone(atlas));
    }
}

/// The persistent GPU buffers. Allocated once at capacity; refilled in place
/// by `write_buffer` (which only reallocates past capacity — never, since the
/// scratch is capacity-bounded).
#[derive(Resource)]
struct FxGpuBuffers {
    vertices: RawBufferVec<FxVertex>,
    indices: RawBufferVec<u32>,
    index_count: u32,
}

impl FxGpuBuffers {
    fn new(render_device: &RenderDevice, vertex_capacity: usize) -> Self {
        let mut vertices = RawBufferVec::new(BufferUsages::VERTEX);
        vertices.reserve(vertex_capacity, render_device);
        let mut indices = RawBufferVec::new(BufferUsages::INDEX);
        indices.reserve(vertex_capacity * 3 / 2, render_device);
        Self {
            vertices,
            indices,
            index_count: 0,
        }
    }
}

fn prepare_fx_buffers(
    mut commands: Commands,
    scratch: Option<Res<FxScratch>>,
    buffers: Option<ResMut<FxGpuBuffers>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(scratch) = scratch else {
        return;
    };
    let Some(mut buffers) = buffers else {
        commands.insert_resource(FxGpuBuffers::new(&render_device, scratch.vertex_capacity));
        return;
    };
    buffers.vertices.clear();
    for vertex in &scratch.vertices {
        buffers.vertices.push(*vertex);
    }
    buffers.indices.clear();
    for index in &scratch.indices {
        buffers.indices.push(*index);
    }
    buffers.index_count = scratch.indices.len() as u32;
    if buffers.index_count > 0 {
        buffers.vertices.write_buffer(&render_device, &render_queue);
        buffers.indices.write_buffer(&render_device, &render_queue);
    }
}

/// View uniforms + atlas texture + sampler, rebuilt only when the view uniform
/// buffer reallocates or the atlas image (re)loads.
#[derive(Resource)]
struct FxBindGroup {
    bind_group: BindGroup,
    view_buffer: BufferId,
    image: AssetId<Image>,
}

fn prepare_fx_bind_group(
    mut commands: Commands,
    pipeline: Option<Res<FxPipeline>>,
    atlas: Option<Res<FxAtlas>>,
    existing: Option<Res<FxBindGroup>>,
    view_uniforms: Res<ViewUniforms>,
    images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
) {
    let (Some(pipeline), Some(atlas)) = (pipeline, atlas) else {
        return;
    };
    let (Some(view_binding), Some(view_buffer)) = (
        view_uniforms.uniforms.binding(),
        view_uniforms.uniforms.buffer().map(|buffer| buffer.id()),
    ) else {
        return;
    };
    let Some(image) = images.get(&atlas.0) else {
        return;
    };
    if existing
        .is_some_and(|existing| existing.view_buffer == view_buffer && existing.image == atlas.0.id())
    {
        return;
    }
    let bind_group = render_device.create_bind_group(
        "fx_bind_group",
        &pipeline.layout,
        &BindGroupEntries::sequential((view_binding, &image.texture_view, &image.sampler)),
    );
    commands.insert_resource(FxBindGroup {
        bind_group,
        view_buffer,
        image: atlas.0.id(),
    });
}

#[derive(Resource)]
struct FxPipeline {
    variants: Variants<RenderPipeline, FxSpecializer>,
    layout: BindGroupLayout,
}

fn init_fx_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let entries = [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: Some(ViewUniform::min_size()),
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
            count: None,
        },
    ];
    // Same entries twice: the device layout binds our bind group, the
    // descriptor lets the pipeline cache resolve an equivalent layout.
    let layout = render_device.create_bind_group_layout("fx_bind_group_layout", &entries);

    let shader = asset_server.load(SHADER_PATH);
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("fx_buffer_pipeline".into()),
        layout: vec![BindGroupLayoutDescriptor::new("fx_bind_group_layout", &entries)],
        vertex: VertexState {
            shader: shader.clone(),
            buffers: vec![VertexBufferLayout {
                array_stride: size_of::<FxVertex>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 12,
                        shader_location: 1,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: 20,
                        shader_location: 2,
                    },
                ],
            }],
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            targets: vec![Some(ColorTargetState {
                // Replaced per-view by the specializer (HDR or not).
                format: TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        // Opaque-pass behavior: depth-tested and depth-written (reverse z),
        // hard alpha handled by `discard` in the shader.
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: default(),
            bias: default(),
        }),
        ..default()
    };

    commands.insert_resource(FxPipeline {
        variants: Variants::new(FxSpecializer, base_descriptor),
        layout,
    });
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct FxKey {
    msaa: Msaa,
    format: TextureFormat,
}

struct FxSpecializer;

impl Specializer<RenderPipeline> for FxSpecializer {
    type Key = FxKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        descriptor.multisample.count = key.msaa.samples();
        if let Some(fragment) = &mut descriptor.fragment
            && let Some(Some(target)) = fragment.targets.first_mut()
        {
            target.format = key.format;
        }
        Ok(key)
    }
}

fn queue_fx(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<FxPipeline>>,
    buffers: Option<Res<FxGpuBuffers>>,
    mut phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = draw_functions.read().id::<DrawFxCommands>();
    let index_count = buffers.map_or(0, |buffers| buffers.index_count);
    // One singleton draw per view; not tied to any main-world entity.
    let main_entity = MainEntity::from(Entity::PLACEHOLDER);

    for (view, msaa) in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        // Binned phases are retained across frames: clear our slot, then
        // re-add it only while there is anything to draw.
        phase.remove(main_entity);
        if index_count == 0 {
            continue;
        }
        let key = FxKey {
            msaa: *msaa,
            format: view.target_format,
        };
        let Ok(pipeline_id) = pipeline.variants.specialize(&pipeline_cache, key) else {
            continue;
        };
        phase.add(
            Opaque3dBatchSetKey {
                draw_function,
                pipeline: pipeline_id,
                material_bind_group_index: None,
                lightmap_slab: None,
                slabs: MeshSlabs::default(),
            },
            Opaque3dBinKey {
                asset_id: AssetId::<Mesh>::invalid().untyped(),
            },
            (Entity::PLACEHOLDER, main_entity),
            InputUniformIndex::default(),
            // NonMesh skips bevy's mesh handling (preprocessing, indirect).
            BinnedRenderPhaseType::NonMesh,
        );
    }
}

type DrawFxCommands = (SetItemPipeline, DrawFxBuffer);

struct DrawFxBuffer;

impl<P: PhaseItem> RenderCommand<P> for DrawFxBuffer {
    type Param = (Option<SRes<FxGpuBuffers>>, Option<SRes<FxBindGroup>>);
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_offset: &'w ViewUniformOffset,
        _entity: Option<()>,
        (buffers, bind_group): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (Some(buffers), Some(bind_group)) = (buffers, bind_group) else {
            return RenderCommandResult::Skip;
        };
        let buffers = buffers.into_inner();
        let (Some(vertex_buffer), Some(index_buffer)) =
            (buffers.vertices.buffer(), buffers.indices.buffer())
        else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(0, &bind_group.into_inner().bind_group, &[view_offset.offset]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..buffers.index_count, 0, 0..1);
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, AssetPlugin::default()))
            .init_asset::<Image>()
            .add_plugins(ParticlesAndTrailsPlugin {
                capacity: 16,
                atlas: "atlas.png".into(),
            });
        // Identity rotation: billboard axes are world X/Y.
        app.world_mut()
            .spawn((Camera3d::default(), Transform::from_xyz(0.0, 0.0, 10.0)));
        app
    }

    fn scratch_data(app: &App) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        let scratch = app.world().resource::<FxScratch>();
        (
            scratch.vertices.iter().map(|v| v.position).collect(),
            scratch.vertices.iter().map(|v| v.uv).collect(),
            scratch.indices.clone(),
        )
    }

    #[test]
    fn particle_lives_in_the_buffer_and_despawns() {
        let mut app = app();
        let sprite = SpriteAnim {
            rect: Rect::from_corners(Vec2::ZERO, Vec2::ONE),
            frames: 4,
            columns: 2,
            fps: 10.0,
        };
        let id = app.world_mut().resource_mut::<Particles>().spawn(Particle {
            sprite,
            position: Vec3::new(1.0, 2.0, 3.0),
            size: Vec2::new(2.0, 4.0),
            ..default()
        });
        app.update();

        let (positions, uvs, indices) = scratch_data(&app);
        assert_eq!(indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(positions[0], [0.0, 0.0, 3.0], "bottom-left corner");
        assert_eq!(positions[2], [2.0, 4.0, 3.0], "top-right corner");
        // Frame 0 of a 2×2 grid over the full atlas is the top-left cell.
        for uv in &uvs[..4] {
            assert!(uv[0] <= 0.5 && uv[1] <= 0.5, "frame 0 uv, got {uv:?}");
        }

        app.world_mut().resource_mut::<Particles>().despawn(id);
        app.update();
        assert!(scratch_data(&app).2.is_empty(), "despawned particle still drawn");
    }

    #[test]
    fn trail_ribbon_emits_quads() {
        let mut app = app();
        app.world_mut().spawn((
            Trail::new(SpriteAnim::FULL, 8, 1000.0, 0.5),
            Transform::from_xyz(0.0, 1.0, 0.0),
        ));
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(5));
        app.update();
        let (_, _, indices) = scratch_data(&app);
        assert!(indices.len() >= 6, "expected a ribbon, got {} indices", indices.len());
    }

    #[test]
    fn overflow_drops_oldest_particles() {
        let mut app = app(); // capacity 16 = 4 quads
        for i in 0..6 {
            app.world_mut().resource_mut::<Particles>().spawn(Particle {
                position: Vec3::X * i as f32,
                ..default()
            });
        }
        app.update();
        let (positions, _, indices) = scratch_data(&app);
        assert_eq!(indices.len(), 4 * 6, "four quads fit");
        assert_eq!(positions[0][0], 5.0 - 0.5, "newest particle wins a slot");
    }
}
