# bevy_particles_and_trails

Every billboard particle and every ribbon trail in your game, in **one draw
call**, on the web. Bevy 0.19. Retro spritesheet FX, the Quake way: a fixed
particle budget, no per-particle entities, no per-frame allocations.

```rust
app.add_plugins(ParticlesAndTrailsPlugin {
    capacity: 8192,                          // vertices: 4/particle, 2/trail point
    atlas: "textures/fx_atlas.png".into(),   // every sprite is a region of this
});

// A particle is a struct in a Vec. Spawning is a push.
particles.spawn(Particle::one_shot(EXPLOSION, hit_point, Vec2::splat(5.0)));

particles.spawn(Particle {
    sprite: DROPLET,
    position, velocity, gravity: 12.0,
    size: Vec2::splat(0.15),
    tint: LinearRgba::rgb(1.0, 0.7, 0.2),
    lifetime: 1.6,
    end_scale: 0.2,      // shrink to 20% over its life
    wiggle_amp: 0.3,     // AE-style noise wiggle
    ..default()
});

// A trail is a component on anything with a GlobalTransform.
commands.spawn((rocket, Trail::new(FIRETRAIL, 64, 60.0, 0.35).with_max_age(0.7)));
```

`cargo run --example fx`

## How

- The CPU rewrites the whole vertex range every frame into a preallocated
  scratch (the `bevy_trail` trick — a bounded memcpy beats growing meshes and
  per-entity extraction). Because everything is re-sent anyway, animation,
  movement, spawning and despawning are plain `Vec` operations.
- The GPU side is a custom `Opaque3d` phase item: two `RawBufferVec`s
  allocated once at capacity and refilled with `write_buffer` — **never
  recreated**. As a per-frame-dirtied `Mesh` asset this allocated a fresh
  staging buffer every frame, and Firefox's WebGPU process OOM'd on the churn.
- Alpha is a hard shader `discard` under 0.5 with depth writes, in the opaque
  pass: unsorted, so buffer order never matters and nothing has to be sorted
  back-to-front. On overflow trails win and the oldest particles drop.

Sprites are grids of equal cells read left-to-right then top-to-bottom
(`columns: 0` = one row, `fps: 0` = static frame 0), all cut out of one atlas.

## The atlas

`scripts/build_fx_atlas.py` packs `fx_sources/*.png` into
`assets/textures/fx_atlas.png` plus a `src/fx_atlas.rs` of region consts, so
adding an effect is dropping in a spritesheet named for its grid:

    fx_sources/explosion.4x4@20.png  ->  pub const EXPLOSION  (4x4 cells, 20 fps)
    fx_sources/droplet.png           ->  pub const DROPLET    (single frame)

`include!("fx_atlas.rs")` next to your particle code and rerun the script after
adding sources. A solid-white cell (`WHITE`) is always packed, for trail
ribbons and flat tinted particles. Nothing forces you to use it — any atlas
plus your own `SpriteAnim` consts works.

## License

MIT OR Apache-2.0
