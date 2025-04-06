use wasm_bindgen::prelude::*;
use rand::{ Rng, SeedableRng };
use rand::rngs::SmallRng;

const GROUND_Y: f32 = 0.0;
const DINO_WIDTH: f32 = 20.0;
const DINO_HEIGHT: f32 = 20.0;
const OBSTACLE_WIDTH: f32 = 20.0;
const OBSTACLE_HEIGHT: f32 = 30.0;
const GRAVITY: f32 = -10.0;
const MAX_JUMP_FORCE: f32 = 30.0; // aumentato da 22.0 a 30.0
const POPULATION_SIZE: usize = 1000;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[wasm_bindgen]
pub struct Obstacle {
    pub x: f32,
    pub base_speed: f32,
}

#[derive(Clone)]
pub struct Dino {
    pub x: f32,
    pub y: f32,
    pub velocity_y: f32,
    pub on_ground: bool,
    pub alive: bool,
    pub score: u32,
    pub time_alive: u32,
}

impl Dino {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y, velocity_y: 0.0, on_ground: true, alive: true, score: 0, time_alive: 0 }
    }

    pub fn update(&mut self, dt: f32) {
        self.time_alive += 1;
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

    pub fn reset(&mut self) {
        self.x = 50.0;
        self.y = GROUND_Y;
        self.velocity_y = 0.0;
        self.on_ground = true;
        self.alive = true;
        self.score = 0;
        self.time_alive = 0;
    }
}

#[derive(Clone)]
pub struct NeuralNet {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub fitness: u32,
}

impl NeuralNet {
    pub fn new(num_inputs: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights = vec![
            -12.0 + rng.gen_range(-1.0..1.0),
            rng.gen_range(-0.5..0.5),
            rng.gen_range(-0.5..0.5)
        ];
        let bias = 6.0 + rng.gen_range(-1.0..1.0);

        Self {
            weights,
            bias,
            fitness: 0,
        }
    }

    pub fn predict(&self, inputs: &[f32]) -> f32 {
        let mut sum = self.bias;
        for (i, w) in self.weights.iter().enumerate() {
            sum += w * inputs[i];
        }
        sigmoid(sum)
    }

    pub fn mutate(&self, mutation_rate: f32, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights = self.weights
            .iter()
            .map(|w| w + rng.gen_range(-mutation_rate..mutation_rate))
            .collect();
        let bias = self.bias + rng.gen_range(-mutation_rate..mutation_rate);
        Self {
            weights,
            bias,
            fitness: 0,
        }
    }
}

#[wasm_bindgen]
pub struct World {
    brains: Vec<NeuralNet>,
    dinos: Vec<Dino>,
    obstacles: Vec<Obstacle>,
    best_index: usize,
    generation: u32,
    fitness_history: Vec<u32>,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut rng = SmallRng::seed_from_u64(42);
        let brains: Vec<NeuralNet> = (0..POPULATION_SIZE)
            .map(|i| NeuralNet::new(3, i as u64))
            .collect();
        let dinos: Vec<Dino> = (0..POPULATION_SIZE).map(|_| Dino::new(50.0, GROUND_Y)).collect();

        Self {
            brains,
            dinos,
            obstacles: vec![
                Obstacle { x: 500.0 + rng.gen_range(-100.0..100.0), base_speed: 50.0 },
                Obstacle { x: 1000.0 + rng.gen_range(-100.0..100.0), base_speed: 50.0 }
            ],
            best_index: 0,
            generation: 0,
            fitness_history: vec![],
        }
    }

    pub fn update(&mut self, dt: f32) {
        let best_score = self.dinos[self.best_index].score as f32;
        let speed_multiplier = 1.0 + best_score / 20.0;

        for (i, dino) in self.dinos.iter_mut().enumerate() {
            if !dino.alive {
                continue;
            }

            dino.update(dt);

            if
                let Some(obs) = self.obstacles
                    .iter()
                    .filter(|o| o.x + OBSTACLE_WIDTH > dino.x)
                    .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
            {
                let dist = obs.x - dino.x;
                let input_distance = ((150.0 - dist) / 150.0).clamp(0.0, 1.0);
                let inputs = vec![
                    input_distance,
                    dino.velocity_y / 10.0,
                    (dino.score as f32) / 100.0
                ];
                let output = self.brains[i].predict(&inputs);

                if dist < 150.0 && dino.on_ground && output > 0.5 {
                    dino.velocity_y = MAX_JUMP_FORCE;
                    dino.on_ground = false;
                }
            }

            for obs in &self.obstacles {
                if Self::is_collision_with(dino.x, dino.y, obs.x, GROUND_Y) {
                    dino.alive = false;
                    self.brains[i].fitness = dino.score * 10 + dino.time_alive;
                    break;
                }
            }
        }

        for obs in &mut self.obstacles {
            obs.x -= obs.base_speed * speed_multiplier * dt;
            if obs.x + OBSTACLE_WIDTH < 0.0 {
                obs.x += 1000.0;
                for dino in self.dinos.iter_mut() {
                    if dino.alive {
                        dino.score += 1;
                    }
                }
            }
        }

        if self.dinos.iter().all(|d| !d.alive) {
            self.evolve();
        }

        self.best_index = self.brains
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.fitness)
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    fn evolve(&mut self) {
        let best = self.brains[self.best_index].clone();

        self.fitness_history.push(best.fitness);
        if self.fitness_history.len() > 100 {
            self.fitness_history.remove(0);
        }

        let mut rng = SmallRng::seed_from_u64(self.generation as u64);

        self.brains = (0..POPULATION_SIZE)
            .map(|i| best.mutate(0.3, (self.generation as u64) + (i as u64)))
            .collect();
        self.dinos = (0..POPULATION_SIZE).map(|_| Dino::new(50.0, GROUND_Y)).collect();
        self.obstacles = vec![
            Obstacle { x: 300.0 + rng.gen_range(-100.0..100.0), base_speed: 50.0 },
            Obstacle { x: 600.0 + rng.gen_range(-100.0..100.0), base_speed: 50.0 }
        ];
        self.generation += 1;
    }

    pub fn get_best_dino_x(&self) -> f32 {
        self.dinos[self.best_index].x
    }

    pub fn get_best_dino_y(&self) -> f32 {
        self.dinos[self.best_index].y
    }

    pub fn get_best_score(&self) -> u32 {
        self.dinos[self.best_index].score
    }

    pub fn get_generation(&self) -> u32 {
        self.generation
    }

    pub fn get_obstacle_count(&self) -> usize {
        self.obstacles.len()
    }

    pub fn get_obstacle_x(&self, index: usize) -> f32 {
        self.obstacles[index].x
    }

    pub fn get_fitness_history(&self) -> Vec<u32> {
        self.fitness_history.clone()
    }

    pub fn get_score(&self) -> u32 {
        self.dinos[0].score
    }

    pub fn get_best_weights(&self) -> Vec<f32> {
        self.brains[self.best_index].weights.clone()
    }

    pub fn get_best_bias(&self) -> f32 {
        self.brains[self.best_index].bias
    }

    pub fn set_best_weights(&mut self, weights: Vec<f32>) {
        self.brains[0].weights = weights;
    }

    pub fn set_best_bias(&mut self, bias: f32) {
        self.brains[0].bias = bias;
    }

    fn is_collision_with(dino_x: f32, dino_y: f32, obs_x: f32, obs_y: f32) -> bool {
        let collision_x = dino_x < obs_x + OBSTACLE_WIDTH && dino_x + DINO_WIDTH > obs_x;
        let collision_y = dino_y < obs_y + OBSTACLE_HEIGHT && dino_y + DINO_HEIGHT > obs_y;
        collision_x && collision_y
    }
}
