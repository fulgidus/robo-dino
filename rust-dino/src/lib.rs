use wasm_bindgen::prelude::*;
use rand::{ Rng, SeedableRng };
use rand::rngs::SmallRng;
use web_sys::console;

const GROUND_Y: f32 = 0.0;
const DINO_WIDTH: f32 = 20.0;
const DINO_HEIGHT: f32 = 20.0;
const OBSTACLE_WIDTH: f32 = 20.0;
const OBSTACLE_HEIGHT: f32 = 30.0;
const GRAVITY: f32 = -45.0;
const MAX_JUMP_FORCE: f32 = 65.0;
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
    pub input_weights: Vec<Vec<f32>>, // [hidden][input]
    pub hidden_biases: Vec<f32>,
    pub output_weights: Vec<f32>, // [hidden]
    pub output_bias: f32,
    pub fitness: u32,
}

impl NeuralNet {
    pub fn new(num_inputs: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let hidden_size = 9;

        let input_weights = (0..hidden_size)
            .map(|_| (0..num_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let hidden_biases = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let output_weights = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let output_bias = rng.gen_range(-1.0..1.0);

        Self {
            input_weights,
            hidden_biases,
            output_weights,
            output_bias,
            fitness: 0,
        }
    }

    pub fn predict(&self, inputs: &[f32]) -> f32 {
        let hidden_activations: Vec<f32> = self.input_weights
            .iter()
            .zip(&self.hidden_biases)
            .map(|(weights, bias)| {
                let sum: f32 = weights
                    .iter()
                    .zip(inputs)
                    .map(|(w, i)| w * i)
                    .sum();
                sigmoid(sum + bias)
            })
            .collect();

        let output: f32 =
            self.output_weights
                .iter()
                .zip(&hidden_activations)
                .map(|(w, h)| w * h)
                .sum::<f32>() + self.output_bias;

        sigmoid(output)
    }

    pub fn mutate(&self, rate: f32, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        let input_weights = self.input_weights
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|w| w + rng.gen_range(-rate..rate))
                    .collect()
            })
            .collect();

        let hidden_biases = self.hidden_biases
            .iter()
            .map(|b| b + rng.gen_range(-rate..rate))
            .collect();

        let output_weights = self.output_weights
            .iter()
            .map(|w| w + rng.gen_range(-rate..rate))
            .collect();

        let output_bias = self.output_bias + rng.gen_range(-rate..rate);

        Self {
            input_weights,
            hidden_biases,
            output_weights,
            output_bias,
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
                Obstacle { x: 300.0 + rng.random_range(-100.0..100.0), base_speed: 50.0 },
                Obstacle { x: 600.0 + rng.random_range(-100.0..100.0), base_speed: 50.0 }
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
                /* web_sys::console::log_1(
                    &format!("🦀: 🔍 Checking collision dino {} vs obs at x = {}", i, obs.x).into()
                ); */
                if Self::is_collision_with(dino.x, dino.y, obs.x, GROUND_Y) {
                    dino.alive = false;
                    web_sys::console::log_1(&format!("🦀: Dino {} morto", i).into());
                    self.brains[i].fitness = dino.score * 10 + dino.time_alive;
                    break;
                }
            }

            web_sys::console::log_1(
                &format!("🦀: Weights for Dino {}: {:?}", i, self.brains[i].input_weights).into()
            );
        }
        let alive_count = self.dinos
            .iter()
            .filter(|d| d.alive)
            .count();
        /* web_sys::console::log_1(&format!("🦀: {} dinos still alive", alive_count).into()); */
        for obs in &mut self.obstacles {
            obs.x -= obs.base_speed * speed_multiplier * dt;
            if obs.x + OBSTACLE_WIDTH < 0.0 {
                let mut rng = SmallRng::seed_from_u64((self.generation as u64) + (obs.x as u64));
                obs.x += 600.0 + rng.random_range(-200.0..200.0);
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
        web_sys::console::log_1(&"🦀: 🌱 Evolving!".into());
        let best = self.brains[self.best_index].clone();
        self.fitness_history.push(best.fitness);
        if self.fitness_history.len() > 100 {
            self.fitness_history.remove(0);
        }

        let seed_base = (self.generation as u64) * 1000;
        let mut new_brains = vec![best.clone()];

        // Mutazioni pesanti del best
        for i in 1..POPULATION_SIZE / 2 {
            new_brains.push(best.mutate(1.5, seed_base + (i as u64)));
        }

        // Individui nuovi random
        for i in POPULATION_SIZE / 2..POPULATION_SIZE {
            new_brains.push(NeuralNet::new(3, seed_base + (i as u64)));
        }

        self.brains = new_brains;
        self.dinos = (0..POPULATION_SIZE).map(|_| Dino::new(50.0, GROUND_Y)).collect();
        let mut rng = SmallRng::seed_from_u64(self.generation as u64);
        self.obstacles = vec![
            Obstacle { x: 600.0 + rng.random_range(-100.0..100.0), base_speed: 50.0 },
            Obstacle { x: 1200.0 + rng.random_range(-100.0..100.0), base_speed: 50.0 }
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

    pub fn get_best_input_weights(&self) -> Vec<f32> {
        self.brains[self.best_index].input_weights.concat()
    }

    pub fn get_best_output_weights(&self) -> Vec<f32> {
        self.brains[self.best_index].output_weights.clone()
    }

    pub fn get_best_bias(&self) -> f32 {
        self.brains[self.best_index].output_bias
    }

    pub fn set_best_weights(&mut self, weights: Vec<f32>) {
        self.brains[0].output_weights = weights;
    }

    pub fn set_best_bias(&mut self, bias: f32) {
        self.brains[0].output_bias = bias;
    }

    fn is_collision_with(dino_x: f32, dino_y: f32, obs_x: f32, obs_y: f32) -> bool {
        let dx = dino_x < obs_x + OBSTACLE_WIDTH && dino_x + DINO_WIDTH > obs_x;
        let dy = dino_y < obs_y + OBSTACLE_HEIGHT && dino_y + DINO_HEIGHT > obs_y;
        let result = dx && dy;
        /* web_sys::console::log_1(
            &format!("🦀: 🔬 collision test: dx={} dy={} => {}", dx, dy, result).into()
        ); */
        result
    }

    #[wasm_bindgen]
    pub fn count_alive(&self) -> usize {
        self.dinos
            .iter()
            .filter(|d| d.alive)
            .count()
    }

    #[wasm_bindgen]
    pub fn get_average_score(&self) -> f32 {
        let total: u32 = self.dinos
            .iter()
            .map(|d| d.score)
            .sum();
        (total as f32) / (self.dinos.len() as f32)
    }

    #[wasm_bindgen]
    pub fn is_alive(&self, index: usize) -> bool {
        self.dinos.get(index).map_or(false, |d| d.alive)
    }
}
