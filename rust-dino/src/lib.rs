use wasm_bindgen::prelude::*;

const GROUND_Y: f32 = 0.0;

const DINO_WIDTH: f32 = 20.0;
const DINO_HEIGHT: f32 = 20.0;

const OBSTACLE_WIDTH: f32 = 20.0;
const OBSTACLE_HEIGHT: f32 = 30.0;

const GRAVITY: f32 = -100.0;
const JUMP_FORCE: f32 = 100.0;

#[wasm_bindgen]
pub struct Obstacle {
    pub x: f32,
    pub base_speed: f32,
}

#[wasm_bindgen]
pub struct Dino {
    pub x: f32,
    pub y: f32,
    pub velocity_y: f32,
    pub on_ground: bool,
}

impl Dino {
    pub fn new(x: f32, y: f32) -> Self {
        Dino {
            x,
            y,
            velocity_y: 0.0,
            on_ground: true,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if !self.on_ground {
            self.velocity_y += GRAVITY * dt;
            self.y += self.velocity_y * dt;

            if self.y <= GROUND_Y {
                self.y = GROUND_Y;
                self.velocity_y = 0.0;
                self.on_ground = true;
            }
        }
    }

    pub fn jump(&mut self) {
        if self.on_ground {
            self.velocity_y = JUMP_FORCE;
            self.on_ground = false;
        }
    }
}

#[wasm_bindgen]
pub struct World {
    dino: Dino,
    obstacles: Vec<Obstacle>,
    score: u32,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        World {
            dino: Dino::new(50.0, GROUND_Y),
            obstacles: vec![
                Obstacle { x: 300.0, base_speed: 50.0 },
                Obstacle { x: 600.0, base_speed: 50.0 }
            ],
            score: 0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.dino.update(dt);
        let dino_x = self.dino.x;
        let dino_y = self.dino.y;

        let speed_multiplier = 1.0 + (self.score as f32) * 0.05;

        for obs in &mut self.obstacles {
            obs.x -= obs.base_speed * speed_multiplier * dt;

            if obs.x + OBSTACLE_WIDTH < 0.0 {
                obs.x += 600.0;
                self.score += 1;
            }

            if World::is_collision_with(dino_x, dino_y, obs.x, GROUND_Y) {
                self.score = 0;
                obs.x = 600.0; // reset ostacolo
            }
        }
    }

    pub fn jump(&mut self) {
        self.dino.jump();
    }

    // Accessors per JS/TS
    pub fn get_dino_x(&self) -> f32 {
        self.dino.x
    }

    pub fn get_dino_y(&self) -> f32 {
        self.dino.y
    }

    pub fn get_score(&self) -> u32 {
        self.score
    }

    pub fn get_obstacle_count(&self) -> usize {
        self.obstacles.len()
    }

    pub fn get_obstacle_x(&self, index: usize) -> f32 {
        self.obstacles[index].x
    }

    // Funzione statica per bounding-box collision
    fn is_collision_with(dino_x: f32, dino_y: f32, obs_x: f32, obs_y: f32) -> bool {
        let collision_x = dino_x < obs_x + OBSTACLE_WIDTH && dino_x + DINO_WIDTH > obs_x;
        let collision_y = dino_y < obs_y + OBSTACLE_HEIGHT && dino_y + DINO_HEIGHT > obs_y;
        collision_x && collision_y
    }
}
