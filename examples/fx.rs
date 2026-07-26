//! Trails, looping emitters and one-shot spritesheet bursts — all of it in a
//! single draw call. Click to fire an explosion at the cursor.

use bevy::prelude::*;
use bevy_particles_and_trails::*;

// Region consts generated from fx_sources/ by scripts/build_fx_atlas.py.
include!("atlas/fx_atlas.rs");

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ParticlesAndTrailsPlugin {
            capacity: 8192,
            atlas: "textures/fx_atlas.png".into(),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (orbit, burst, sparks))
        .run();
}

#[derive(Component)]
struct Orbiter(f32);

fn setup(mut commands: Commands, mut particles: ResMut<Particles>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 14.0).looking_at(Vec3::Y * 2.0, Vec3::Y),
    ));

    // Three ribbons chasing each other around the origin.
    for (i, tint) in [LinearRgba::rgb(1.0, 0.4, 0.1), LinearRgba::rgb(0.2, 0.8, 1.0), LinearRgba::rgb(1.0, 1.0, 0.3)]
        .into_iter()
        .enumerate()
    {
        let mut trail = Trail::new(WHITE, 64, 60.0, 0.35).with_max_age(0.7);
        trail.tint = tint;
        commands.spawn((
            Orbiter(i as f32 * std::f32::consts::TAU / 3.0),
            Transform::default(),
            trail,
        ));
    }

    // A looping animation that never despawns.
    particles.spawn(Particle {
        sprite: FIREWOOSH,
        position: Vec3::new(0.0, 2.0, 0.0),
        size: Vec2::splat(3.0),
        ..default()
    });
}

fn orbit(time: Res<Time>, mut orbiters: Query<(&Orbiter, &mut Transform)>) {
    for (orbiter, mut transform) in &mut orbiters {
        let t = time.elapsed_secs() * 1.5 + orbiter.0;
        transform.translation = Vec3::new(t.cos() * 5.0, 2.0 + (t * 2.0).sin() * 1.5, t.sin() * 5.0);
    }
}

/// One-shot: plays the sheet once at its own fps, then removes itself.
fn burst(buttons: Res<ButtonInput<MouseButton>>, mut particles: ResMut<Particles>, time: Res<Time>) {
    if buttons.just_pressed(MouseButton::Left) {
        let t = time.elapsed_secs();
        particles.spawn(Particle::one_shot(
            EXPLOSION,
            Vec3::new(t.sin() * 4.0, 2.0, t.cos() * 4.0),
            Vec2::splat(5.0),
        ));
    }
}

/// Thousands of cheap gravity sprites — the retro particle budget.
fn sparks(time: Res<Time>, mut particles: ResMut<Particles>) {
    let t = time.elapsed_secs();
    for i in 0..8 {
        let angle = t * 7.0 + i as f32;
        particles.spawn(Particle {
            sprite: DROPLET,
            position: Vec3::Y * 0.2,
            velocity: Vec3::new(angle.cos() * 4.0, 8.0, angle.sin() * 4.0),
            gravity: 12.0,
            size: Vec2::splat(0.15),
            tint: LinearRgba::rgb(1.0, 0.7, 0.2),
            lifetime: 1.6,
            end_scale: 0.2,
            ..default()
        });
    }
}
