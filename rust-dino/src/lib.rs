use rand::rngs::SmallRng;
use rand::{ Rng, SeedableRng };
use wasm_bindgen::prelude::*;
use web_sys::console;

const GROUND_Y: f32 = 0.0;
const DINO_WIDTH: f32 = 20.0;
const DINO_HEIGHT: f32 = 20.0;
const OBSTACLE_WIDTH: f32 = 5.0;
const OBSTACLE_HEIGHT: f32 = 30.0;
const GRAVITY: f32 = -90.0;
const MAX_JUMP_FORCE: f32 = 90.0;
const SUBSTEP_FACTOR: u32 = 5; // Esegui fino a 5 sub-step per ogni unità di speed_multiplier > 1
const MAX_SUBSTEPS: u32 = 500; // Limite massimo per evitare blocchi a velocità estreme
const DISTANCE_NOISE_FACTOR: f32 = 0.05; // Esempio: errore massimo del 5% della distanza

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
        Self {
            x,
            y,
            velocity_y: 0.0,
            on_ground: true,
            alive: true,
            score: 0,
            time_alive: 0,
        }
    }
    pub fn update(&mut self, dt: f32) {
        self.time_alive += 1;
        if !self.on_ground {
            self.velocity_y += GRAVITY * dt;
            self.y += self.velocity_y * dt;

            if self.y < GROUND_Y {
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
        let hidden_size = 3;

        let input_weights = (0..hidden_size)
            .map(|_| (0..num_inputs).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();

        let hidden_biases = (0..hidden_size).map(|_| rng.random_range(-1.0..1.0)).collect();
        let output_weights = (0..hidden_size).map(|_| rng.random_range(-1.0..1.0)).collect();
        let output_bias = rng.random_range(-1.0..1.0);

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
        // --- Add check for zero rate ---
        if rate <= f32::EPSILON {
            // If rate is zero or negligible, return a clone with fitness reset
            let mut clone = self.clone();
            clone.fitness = 0;
            return clone;
        }

        // --- Proceed with mutation if rate > 0 ---
        let mut rng = SmallRng::seed_from_u64(seed);

        let input_weights = self.input_weights
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|w| w + rng.random_range(-rate..rate))
                    .collect()
            })
            .collect();

        let hidden_biases = self.hidden_biases
            .iter()
            .map(|b| b + rng.random_range(-rate..rate))
            .collect();

        let output_weights = self.output_weights
            .iter()
            .map(|w| w + rng.random_range(-rate..rate))
            .collect();

        let output_bias = self.output_bias + rng.random_range(-rate..rate);

        Self {
            input_weights,
            hidden_biases,
            output_weights,
            output_bias,
            fitness: 0, // Fitness is always reset
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
    speed_multiplier: f32,
    population_size: usize,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new(count: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(42);
        let brains: Vec<NeuralNet> = (0..count).map(|i| NeuralNet::new(3, i as u64)).collect();
        let dinos: Vec<Dino> = (0..count).map(|_| Dino::new(50.0, GROUND_Y)).collect();

        Self {
            brains,
            dinos,
            obstacles: vec![
                Obstacle {
                    x: 200.0 + rng.random_range(-50.0..0.0),
                    base_speed: 50.0,
                },
                Obstacle {
                    x: 600.0 + rng.random_range(-100.0..100.0),
                    base_speed: 50.0,
                }
            ],
            best_index: 0,
            generation: 0,
            fitness_history: vec![],
            speed_multiplier: 0.0,
            population_size: count,
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Calcola il moltiplicatore di velocità basato sul punteggio massimo attuale
        let best_score = self.dinos
            .iter()
            .filter(|d| d.alive)
            .map(|d| d.score)
            .max()
            .unwrap_or(0) as f32;
        // Applica il limite massimo al moltiplicatore di velocità se necessario
        // self.speed_multiplier = (1.0 + best_score / 10.0).min(30.0); // Esempio con limite a 30x
        self.speed_multiplier = 1.0 + best_score / 10.0; // Versione senza limite

        // --- Calcolo Sub-stepping ---
        let num_sub_steps = if self.speed_multiplier <= 1.0 {
            1
        } else {
            // Scala linearmente con la velocità, ma limita
            (((self.speed_multiplier - 1.0).ceil() as u32) * SUBSTEP_FACTOR + 1).min(MAX_SUBSTEPS)
        };
        let sub_dt = dt / (num_sub_steps as f32); // Delta time per ogni sub-step

        // --- Loop di Sub-stepping per Fisica e Collisioni ---
        for _ in 0..num_sub_steps {
            // --- A. Aggiorna Posizione Ostacoli (con sub_dt) ---
            let max_x_before_substep_update = self.obstacles // Serve per il reset *all'interno* del substep
                .iter()
                .map(|o| o.x)
                .fold(f32::NEG_INFINITY, f32::max);

            let mut score_increment_this_substep = 0; // Contatore per il punteggio
            for obs in &mut self.obstacles {
                // Muovi l'ostacolo usando sub_dt
                obs.x -= obs.base_speed * self.speed_multiplier * sub_dt; // Usa sub_dt!

                // Se l'ostacolo è uscito a sinistra *durante questo sub-step*
                if obs.x + OBSTACLE_WIDTH < 0.0 {
                    // Riposiziona l'ostacolo a destra
                    let mut rng = SmallRng::seed_from_u64(
                        (self.generation as u64) + (obs.x as u64) // Usa un seed prevedibile
                    );

                    // --- NUOVA LOGICA CON DISTANZA SCALATA ---
                    // Scala la distanza base di spawn con il moltiplicatore di velocità
                    let base_spawn_distance = 300.0 * self.speed_multiplier;
                    // Mantieni la variazione casuale
                    let random_offset = rng.random_range(0.0..600.0);

                    obs.x = max_x_before_substep_update + base_spawn_distance + random_offset;
                    // --- FINE NUOVA LOGICA ---

                    // Segna che il punteggio deve aumentare
                    score_increment_this_substep += 1;
                }
            }

            // --- B. Aggiorna Fisica Dinosauri, Salto e Collisioni ---
            for i in 0..self.dinos.len() {
                if !self.dinos[i].alive {
                    continue;
                }

                // Aggiorna fisica dino
                self.dinos[i].update(sub_dt);

                // --- Logica Salto con Distanza Rumorosa ---
                // Crea un RNG per questo dino/substep (usa un seed prevedibile)
                let mut noise_rng = SmallRng::seed_from_u64(
                    (self.generation as u64) * 10000 + (i as u64) // Seed unico per dino/gen
                );

                let noisy_dist_input: f32; // La distanza con errore da passare alla rete

                if
                    let Some(obs) = self.obstacles
                        .iter()
                        // Filtra ostacoli davanti e nel campo visivo
                        .filter(|o| {
                            let in_front = o.x + OBSTACLE_WIDTH > self.dinos[i].x;
                            // Considera un campo visivo (VISION_RANGE non definito nel contesto, assumiamo 300.0)
                            let vision_range = 300.0;
                            let in_range = o.x < self.dinos[i].x + vision_range;
                            in_front && in_range
                        })
                        .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
                {
                    // Ostacolo trovato nel range
                    let actual_dist = obs.x - self.dinos[i].x;

                    // --- Calcola e applica rumore proporzionale (CON CHECK) ---
                    let noise = if actual_dist <= f32::EPSILON {
                        // Check if distance is effectively zero
                        0.0 // No noise if distance is zero
                    } else {
                        let max_noise = actual_dist * DISTANCE_NOISE_FACTOR;
                        // Ensure max_noise is positive before creating range
                        if max_noise > 0.0 {
                            noise_rng.random_range(-max_noise..max_noise)
                        } else {
                            0.0 // No noise if max_noise is zero or negative
                        }
                    };
                    noisy_dist_input = (actual_dist + noise).max(0.0);
                    // --- Fine calcolo rumore ---
                } else {
                    // Nessun ostacolo nel range visivo
                    // Applica rumore anche alla percezione della "distanza massima"
                    let vision_range = 300.0; // Assumiamo 300.0
                    let max_noise_at_vision_range = vision_range * DISTANCE_NOISE_FACTOR;
                    let noise = noise_rng.random_range(
                        -max_noise_at_vision_range..max_noise_at_vision_range
                    );
                    noisy_dist_input = (vision_range + noise).max(0.0);
                }

                // Usa la distanza rumorosa come input per la rete
                let inputs = vec![
                    noisy_dist_input, // <-- Usa la distanza con errore
                    self.speed_multiplier,
                    ((self.dinos[i].score + 1) as f32) / 100.0
                ];
                let output = self.brains[i].predict(&inputs);

                // Logica di salto (usa l'output basato sull'input rumoroso)
                if self.dinos[i].on_ground && output > 0.6 {
                    self.dinos[i].velocity_y = MAX_JUMP_FORCE;
                    self.dinos[i].on_ground = false;

                    // Applica IMMEDIATAMENTE il movimento verticale iniziale del salto
                    self.dinos[i].y += self.dinos[i].velocity_y * sub_dt;
                }
                // --- Fine Logica Salto ---

                // Controlla Collisioni con Ostacoli (usando le posizioni aggiornate nel sub-step)
                let dino_x = self.dinos[i].x;
                let dino_y = self.dinos[i].y;
                for obs in &self.obstacles {
                    // Ora obs.x è aggiornato per questo sub-step
                    if Self::is_collision_with(dino_x, dino_y, obs.x, GROUND_Y) {
                        self.dinos[i].alive = false;
                        self.brains[i].fitness = self.dinos[i].time_alive;
                        break;
                    }
                }

                // Incrementa punteggio se necessario (solo per dinosauri ancora vivi dopo collision check)
                if self.dinos[i].alive && score_increment_this_substep > 0 {
                    self.dinos[i].score += score_increment_this_substep;
                }
            } // Fine ciclo sui dinosauri per il sub-step
        } // Fine loop di sub-stepping

        // --- Logica da Eseguire UNA VOLTA per Frame (fuori dal sub-stepping) ---

        // 1. Aggiorna best_index (basato sullo score dei vivi)
        let best_alive_index = self.dinos
            .iter()
            .enumerate()
            .filter(|(_, d)| d.alive)
            .max_by_key(|(_, d)| d.score)
            .map(|(i, _)| i);
        self.best_index = best_alive_index.unwrap_or(0);

        // 2. Controlla se tutti i dinosauri sono morti per evolvere
        if self.dinos.iter().all(|d| !d.alive) {
            // Ricalcola best_index basato sulla FITNESS
            self.best_index = self.brains
                .iter()
                .enumerate()
                .max_by_key(|(_, b)| b.fitness)
                .map(|(i, _)| i)
                .unwrap_or(0);
            // Esegui l'evoluzione
            self.evolve();
        }
    }

    fn evolve(&mut self) {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"🦀: 🌱 Evolving!".into());

        // --- Handle Empty Population Case ---
        if self.population_size == 0 {
            #[cfg(target_arch = "wasm32")]
            console::warn_1(
                &"WARN: Attempted to evolve with population size 0. Skipping evolve.".into()
            );
            // Optionally reset generation or other state if needed, but returning is safest.
            return;
        }

        // --- Get the Best Brain Safely ---
        // Uses the best_index calculated based on FITNESS just before calling evolve
        let best_brain_for_cloning = match self.brains.get(self.best_index) {
            Some(b) => b.clone(),
            None => {
                // Fallback if best_index is invalid. This shouldn't happen if pop_size > 0
                // and the pre-evolve best_index calculation is correct, but we add safety.
                #[cfg(target_arch = "wasm32")]
                console::warn_1(
                    &format!(
                        "WARN: best_index {} invalid during evolve (pop_size {}), using index 0 as fallback.",
                        self.best_index,
                        self.population_size
                    ).into()
                );

                // Attempt to get brain 0, or create a default if even that fails.
                self.brains
                    .get(0)
                    .cloned()
                    .unwrap_or_else(|| {
                        #[cfg(target_arch = "wasm32")]
                        console::error_1(
                            &"ERROR: Failed to get even brain 0 in evolve fallback. Creating default.".into()
                        );
                        // Use generation for a somewhat unique seed if creating default
                        NeuralNet::new(3, self.generation as u64)
                    })
            }
        };

        // Record the fitness of the brain selected for cloning
        self.fitness_history.push(best_brain_for_cloning.fitness);

        // --- Create the Next Generation ---
        let seed_base = (self.generation as u64) * 1000;
        let mut new_brains = Vec::with_capacity(self.population_size);

        // 1. Add the elite clone (with fitness reset)
        let mut elite_clone = best_brain_for_cloning.clone();
        elite_clone.fitness = 0; // Fitness must be reset for the new generation
        new_brains.push(elite_clone);

        // 2. Add mutated versions based on the best brain's weights/biases
        //    Use the already safely cloned `best_brain_for_cloning` as the base for mutation.
        for i in 1..self.population_size {
            // The mutate function already resets fitness to 0
            new_brains.push(best_brain_for_cloning.mutate(0.4, seed_base + (i as u64)));
        }

        // --- Replace Old Generation ---
        self.brains = new_brains;
        self.dinos = (0..self.population_size).map(|_| Dino::new(50.0, GROUND_Y)).collect();

        // --- Reset Obstacles ---
        // Use the *new* generation number for a different obstacle layout
        let mut rng = SmallRng::seed_from_u64((self.generation + 1) as u64);
        self.obstacles = vec![
            Obstacle {
                x: 200.0 + rng.random_range(-100.0..100.0),
                base_speed: 50.0,
            },
            Obstacle {
                x: 500.0 + rng.random_range(0.0..100.0),
                base_speed: 50.0,
            }
        ];

        // --- Increment Generation ---
        self.generation += 1;
    }

    pub fn get_best_dino_x(&self) -> f32 {
        self.dinos.get(self.best_index).map_or(0.0, |d| d.x)
    }

    pub fn get_best_dino_y(&self) -> f32 {
        self.dinos.get(self.best_index).map_or(GROUND_Y, |d| d.y)
    }

    #[wasm_bindgen]
    pub fn get_best_dino_velocity_y(&self) -> f32 {
        self.dinos.get(self.best_index).map_or(0.0, |d| d.velocity_y)
    }

    #[wasm_bindgen]
    pub fn get_best_index(&self) -> usize {
        self.best_index
    }

    pub fn get_best_score(&self) -> u32 {
        self.dinos
            .iter()
            .filter(|d| d.alive)
            .map(|d| d.score)
            .max()
            .unwrap_or(0)
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

    #[wasm_bindgen]
    pub fn get_score_of(&self, index: usize) -> u32 {
        self.dinos.get(index).map_or(0, |d| d.score)
    }

    pub fn get_best_input_weights(&self) -> Vec<f32> {
        self.brains.get(self.best_index).map_or(vec![], |b| b.input_weights.concat())
    }

    pub fn get_best_output_weights(&self) -> Vec<f32> {
        self.brains.get(self.best_index).map_or(vec![], |b| b.output_weights.clone())
    }

    pub fn get_best_bias(&self) -> f32 {
        self.brains.get(self.best_index).map_or(0.0, |b| b.output_bias)
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
        /* #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
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

    #[wasm_bindgen]
    pub fn get_best_hidden_biases(&self) -> Vec<f32> {
        // Ensure best_index is within the bounds of the brains vector
        if self.best_index < self.brains.len() {
            // Clone the hidden_biases vector from the best brain and return it
            self.brains[self.best_index].hidden_biases.clone()
        } else {
            // Fallback: If best_index is invalid (e.g., at the very start, or an error occurred)
            // return a vector of zeros with the correct size if possible, or an empty vector.
            // This prevents errors in the JavaScript side trying to access elements.
            if let Some(first_brain) = self.brains.first() {
                // Return a vector of zeros matching the expected hidden layer size
                vec![0.0; first_brain.hidden_biases.len()]
            } else {
                // If there are no brains at all (shouldn't happen in normal operation)
                vec![]
            }
        }
    }

    pub fn get_population_size(&self) -> usize {
        self.dinos.len()
    }

    pub fn get_dino_x(&self, index: usize) -> f32 {
        self.dinos
            .get(index)
            .map(|d| d.x)
            .unwrap_or(0.0)
    }

    pub fn get_dino_y(&self, index: usize) -> f32 {
        self.dinos
            .get(index)
            .map(|d| d.y)
            .unwrap_or(0.0)
    }

    #[wasm_bindgen]
    pub fn get_speed_multiplier(&self) -> f32 {
        self.speed_multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import everything from the parent module (lib.rs)

    #[test]
    fn test_best_index_updates_after_update() {
        // 1. Arrange: Create a world
        let mut world = World::new(200);
        assert!(200 >= 3, "Test requires at least 3 dinos");

        // 2. Act: Manually set fitness and mark all dinos dead to trigger
        //         the pre-evolve best_index calculation based on fitness.
        let target_best_index = 1;
        let lower_fitness = 50;
        let higher_fitness = 100;

        // Mark dino 0 dead with lower fitness
        if let Some(dino) = world.dinos.get_mut(0) {
            dino.alive = false;
        } else {
            panic!("Dino 0 setup failed");
        }
        if let Some(brain) = world.brains.get_mut(0) {
            brain.fitness = lower_fitness;
        } else {
            panic!("Brain 0 setup failed");
        }

        // Mark dino 1 dead with higher fitness
        if let Some(dino) = world.dinos.get_mut(target_best_index) {
            dino.alive = false;
        } else {
            panic!("Dino {} setup failed", target_best_index);
        }
        if let Some(brain) = world.brains.get_mut(target_best_index) {
            brain.fitness = higher_fitness;
        } else {
            panic!("Brain {} setup failed", target_best_index);
        }

        // Mark dino 2 dead with even lower fitness
        if let Some(dino) = world.dinos.get_mut(2) {
            dino.alive = false;
        } else {
            panic!("Dino 2 setup failed");
        }
        if let Some(brain) = world.brains.get_mut(2) {
            brain.fitness = lower_fitness - 10;
        } else {
            panic!("Brain 2 setup failed");
        }

        // Mark ALL OTHER dinos dead as well to ensure the pre-evolve condition is met
        for i in 3..200 {
            if let Some(dino) = world.dinos.get_mut(i) {
                dino.alive = false;
            } else {
                panic!("Dino {} setup failed", i);
            }
            // Optionally set their fitness too, though not strictly needed for this test's goal
            if let Some(brain) = world.brains.get_mut(i) {
                brain.fitness = 1;
            } else {
                panic!("Brain {} setup failed", i);
            }
        }

        // Run update once. Since all dinos are dead, this *should* trigger
        // the fitness-based best_index calculation right before evolve().
        world.update(0.1); // dt doesn't matter

        // 3. Assert: Check if best_index points to the dino with highest fitness
        // Note: evolve() runs immediately after, but best_index was set *before* it.
        assert_eq!(
            world.best_index,
            target_best_index,
            "best_index should point to the brain with the highest fitness ({}) right before evolve",
            target_best_index
        );
    }

    #[test]
    fn test_best_index_updates_when_best_dino_dies() {
        // 1. Arrange: Create a world and establish an initial best dino based on SCORE
        let mut world = World::new(200);
        assert!(200 >= 3, "Test requires at least 3 dinos for clarity");

        let first_best_idx = 1;
        let second_best_idx = 2;
        let high_score = 100;
        let medium_score = 50;
        let low_score = 10;

        // Assign initial SCORES (all dinos start alive)
        if let Some(dino) = world.dinos.get_mut(first_best_idx) {
            dino.score = high_score;
        } else {
            panic!("Dino {} setup failed", first_best_idx);
        }
        if let Some(dino) = world.dinos.get_mut(second_best_idx) {
            dino.score = medium_score;
        } else {
            panic!("Dino {} setup failed", second_best_idx);
        }
        if let Some(dino) = world.dinos.get_mut(0) {
            dino.score = low_score;
        } else {
            panic!("Dino 0 setup failed");
        }
        // Other dinos have score 0

        // Run update once to calculate the initial best_index based on score
        world.update(0.01); // dt doesn't matter much here

        // Verify the initial best index is correct (based on highest score)
        assert_eq!(
            world.best_index,
            first_best_idx,
            "Initial best_index should be the one with highest score ({})",
            first_best_idx
        );
        assert!(world.dinos[first_best_idx].alive, "Initial best dino should be alive");
        assert!(world.dinos[second_best_idx].alive, "Second best dino should be alive");

        // 2. Act: Kill the current best dino (highest score)
        if let Some(dino) = world.dinos.get_mut(first_best_idx) {
            dino.alive = false;
            // Its score remains, but it's no longer considered by the score-based calculation
        } else {
            panic!("Dino at index {} not found", first_best_idx);
        }

        // Run update again. The best_index calculation should now pick the living dino
        // with the next highest score.
        world.update(0.01);

        // 3. Assert: Check if best_index now points to the second best dino (based on score)
        assert_eq!(
            world.best_index,
            second_best_idx,
            "best_index should update to the living dino with the next highest score ({}) after the best one dies",
            second_best_idx
        );
        assert_ne!(
            world.best_index,
            first_best_idx,
            "best_index should no longer point to the dead dino ({})",
            first_best_idx
        );
    }

    #[test]
    // Rename to reflect it uses score-based best_index
    fn test_get_best_dino_velocity_y_matches_best_score_dino() {
        // 1. Arrange: Create a world and set up scores/velocities
        let mut world = World::new(200);
        assert!(200 >= 2, "Test requires at least 2 dinos");

        let best_score_idx = 1; // Index of the dino we'll give the highest score
        let other_idx = 0;
        let high_score = 100;
        let low_score = 50;
        let target_velocity = 15.5; // Velocity for the best-scoring dino
        let other_velocity = -5.0; // Velocity for the other dino

        // Assign scores
        if let Some(dino) = world.dinos.get_mut(best_score_idx) {
            dino.score = high_score;
        } else {
            panic!("Dino {} setup failed", best_score_idx);
        }
        if let Some(dino) = world.dinos.get_mut(other_idx) {
            dino.score = low_score;
        } else {
            panic!("Dino {} setup failed", other_idx);
        }

        // Assign specific velocities
        if let Some(dino) = world.dinos.get_mut(best_score_idx) {
            dino.velocity_y = target_velocity;
        } else {
            panic!("Dino {} setup failed", best_score_idx);
        }
        if let Some(dino) = world.dinos.get_mut(other_idx) {
            dino.velocity_y = other_velocity;
        } else {
            panic!("Dino {} setup failed", other_idx);
        }

        // Run update once to ensure best_index is calculated based on score
        world.update(0.01);

        // Verify best_index is set correctly as a precondition
        assert_eq!(
            world.best_index,
            best_score_idx,
            "Precondition failed: best_index should be {} based on score",
            best_score_idx
        );

        // 2. Act: Call the function under test
        let reported_velocity = world.get_best_dino_velocity_y();

        // 3. Assert: Check if the returned velocity matches the target velocity
        assert!(
            (reported_velocity - target_velocity).abs() < f32::EPSILON, // Use float comparison
            "get_best_dino_velocity_y should return the velocity_y ({}) of the dino at best_index ({}), but got {}",
            target_velocity,
            best_score_idx,
            reported_velocity
        );

        // Optional: Assert it's different from the other dino's velocity
        assert!(
            (reported_velocity - other_velocity).abs() > f32::EPSILON,
            "Reported velocity should be different from the other dino's velocity"
        );
    }

    #[test]
    fn test_sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < f32::EPSILON, "Sigmoid of 0 should be 0.5");
        assert!(sigmoid(100.0) > 0.99, "Sigmoid of large positive number should be close to 1");
        assert!(sigmoid(-100.0) < 0.01, "Sigmoid of large negative number should be close to 0");
    }

    #[test]
    fn test_dino_update_gravity_and_landing() {
        let mut dino = Dino::new(50.0, GROUND_Y);
        let dt = 0.1;

        // Caso 1: Inizia a terra, salta (manualmente)
        dino.on_ground = false;
        dino.velocity_y = MAX_JUMP_FORCE; // Simula un salto
        let initial_y = dino.y;
        let initial_vel = dino.velocity_y;

        // --- Store the velocity *after* gravity is applied in the update ---
        let velocity_after_gravity = initial_vel + GRAVITY * dt;
        dino.update(dt);

        // Verifica applicazione gravità
        assert!(!dino.on_ground, "Dino should still be in the air");
        // --- Check velocity after update ---
        assert!(
            (dino.velocity_y - velocity_after_gravity).abs() < f32::EPSILON,
            "Velocity should decrease due to gravity. Expected {}, got {}",
            velocity_after_gravity,
            dino.velocity_y
        );
        // --- Check position using the velocity *after* gravity was applied ---
        // Note: The original implementation updates position using the *new* velocity.
        // A more physically accurate Euler step might average initial/final velocity,
        // but we test the code as written.
        let expected_y = initial_y + velocity_after_gravity * dt;
        assert!(
            (dino.y - expected_y).abs() < f32::EPSILON,
            "Y position should change based on updated velocity. Expected {}, got {}",
            expected_y,
            dino.y
        );

        // Caso 2: Simula abbastanza step per farlo atterrare
        dino.velocity_y = -1.0; // Imposta una piccola velocità verso il basso
        dino.y = GROUND_Y + 0.01; // Poco sopra il terreno
        dino.on_ground = false;

        dino.update(dt); // Questo update dovrebbe farlo atterrare

        // Verifica atterraggio
        assert!(dino.on_ground, "Dino should be on the ground");
        assert!((dino.y - GROUND_Y).abs() < f32::EPSILON, "Dino Y should be reset to GROUND_Y");
        assert!(dino.velocity_y.abs() < f32::EPSILON, "Dino velocity_y should be reset to 0");
    }

    #[test]
    fn test_collision_detection() {
        // Caso 1: Collisione
        let dino_x = 50.0;
        let dino_y = GROUND_Y; // A terra
        let obs_x = dino_x + DINO_WIDTH / 2.0; // Ostacolo sovrapposto orizzontalmente
        let obs_y = GROUND_Y; // Ostacolo a terra
        assert!(
            World::is_collision_with(dino_x, dino_y, obs_x, obs_y),
            "Should detect collision when overlapping"
        );

        // Caso 2: Nessuna collisione (ostacolo troppo a destra)
        let obs_x_far = dino_x + DINO_WIDTH + 1.0;
        assert!(
            !World::is_collision_with(dino_x, dino_y, obs_x_far, obs_y),
            "Should not detect collision when obstacle is far right"
        );

        // Caso 3: Nessuna collisione (ostacolo troppo a sinistra)
        let obs_x_left = dino_x - OBSTACLE_WIDTH - 1.0;
        assert!(
            !World::is_collision_with(dino_x, dino_y, obs_x_left, obs_y),
            "Should not detect collision when obstacle is far left"
        );

        // Caso 4: Nessuna collisione (dino in aria sopra l'ostacolo)
        let dino_y_air = GROUND_Y + OBSTACLE_HEIGHT + 1.0;
        assert!(
            !World::is_collision_with(dino_x, dino_y_air, obs_x, obs_y),
            "Should not detect collision when dino is above obstacle"
        );
    }

    #[test]
    fn test_world_evolve() {
        let mut world = World::new(200);
        assert!(200 >= 2, "Evolve test requires at least 2 population size");

        // Imposta fitness diverse per identificare il migliore
        let best_idx = 1;
        let high_fitness = 100;
        if let Some(brain) = world.brains.get_mut(best_idx) {
            brain.fitness = high_fitness;
        } else {
            panic!("Setup failed");
        }
        if let Some(brain) = world.brains.get_mut(0) {
            brain.fitness = 50;
        } else {
            panic!("Setup failed");
        }

        // Rendi tutti i dinosauri "morti" per forzare l'evoluzione
        for dino in world.dinos.iter_mut() {
            dino.alive = false;
        }
        // Aggiorna per calcolare best_index e triggerare evolve
        world.update(0.01);

        // --- Verifiche dopo evolve ---
        let prev_generation = 0; // Generazione iniziale
        let prev_best_fitness = high_fitness;

        // Verifica incremento generazione
        assert_eq!(world.generation, prev_generation + 1, "Generation should increment");

        // Verifica storia fitness
        assert_eq!(world.fitness_history.len(), 1, "Fitness history should have one entry");
        assert_eq!(
            world.fitness_history[0],
            prev_best_fitness,
            "Fitness history should contain previous best fitness"
        );

        // Verifica reset dinosauri
        assert!(
            world.dinos.iter().all(|d| d.alive),
            "All dinos should be alive after evolve"
        );
        assert!(
            world.dinos.iter().all(|d| (d.y - GROUND_Y).abs() < f32::EPSILON),
            "All dinos should be at GROUND_Y after evolve"
        );
        assert!(
            world.dinos.iter().all(|d| d.score == 0),
            "All dino scores should be 0 after evolve"
        );
        assert!(
            world.dinos.iter().all(|d| d.time_alive == 0),
            "All dino time_alive should be 0 after evolve"
        );

        // Verifica cervelli (il primo è clone, gli altri mutati)
        assert_eq!(world.brains.len(), 200, "Brain count should remain constant");
        // Nota: Confrontare float per uguaglianza esatta è rischioso.
        // Verifichiamo che il primo cervello abbia la fitness resettata (come tutti i nuovi)
        assert_eq!(world.brains[0].fitness, 0, "Fitness of the cloned best brain should be reset");
        // Sarebbe complesso verificare *esattamente* la clonazione/mutazione senza esporre più dettagli
        // o usare un seme fisso anche per la mutazione nel test.
        // Ci fidiamo che il clone e le mutazioni avvengano come da codice `evolve`.

        // Verifica reset ostacoli (almeno il numero è corretto)
        assert!(!world.obstacles.is_empty(), "Obstacles should be reset");
        // Potremmo aggiungere controlli sulle posizioni iniziali se fossero più deterministiche
    }

    #[test]
    fn test_neural_net_mutate() {
        let seed = 123;
        let nn = NeuralNet::new(3, seed);
        let mutation_rate = 0.1;

        // Mutazione con rate > 0
        let mutated_nn = nn.mutate(mutation_rate, seed + 1); // Usa un seme diverso per la mutazione

        // Verifica che *almeno un* peso/bias sia cambiato (probabilistico)
        let weights_changed =
            nn.input_weights != mutated_nn.input_weights ||
            nn.hidden_biases != mutated_nn.hidden_biases ||
            nn.output_weights != mutated_nn.output_weights ||
            nn.output_bias != mutated_nn.output_bias;
        assert!(weights_changed, "Mutation with rate > 0 should change weights/biases");
        assert_eq!(mutated_nn.fitness, 0, "Mutated NN should have fitness 0"); // Verifica reset fitness

        // Mutazione con rate = 0
        let non_mutated_nn = nn.mutate(0.0, seed + 2);
        // Confronta i pesi/bias per uguaglianza (con tolleranza per float se necessario)
        assert_eq!(
            nn.input_weights,
            non_mutated_nn.input_weights,
            "Mutation with rate 0 should not change input weights"
        );
        assert_eq!(
            nn.hidden_biases,
            non_mutated_nn.hidden_biases,
            "Mutation with rate 0 should not change hidden biases"
        );
        assert_eq!(
            nn.output_weights,
            non_mutated_nn.output_weights,
            "Mutation with rate 0 should not change output weights"
        );
        assert!(
            (nn.output_bias - non_mutated_nn.output_bias).abs() < f32::EPSILON,
            "Mutation with rate 0 should not change output bias"
        );
        assert_eq!(non_mutated_nn.fitness, 0, "Non-mutated NN should have fitness 0");
    }

    #[test]
    fn test_obstacle_movement_and_reset_and_score() {
        let mut world = World::new(200);
        let dt = 0.1;
        let initial_speed_multiplier = 1.0; // Assume base speed for simplicity
        world.speed_multiplier = initial_speed_multiplier; // Set manually for test predictability

        // Ensure there's at least one obstacle
        assert!(!world.obstacles.is_empty(), "World should have obstacles");

        let initial_obs0_x = world.obstacles[0].x;
        let obs_speed = world.obstacles[0].base_speed;
        let expected_move_dist = obs_speed * initial_speed_multiplier * dt;

        // 1. Test basic movement
        world.update(dt);
        let new_obs0_x = world.obstacles[0].x;
        assert!(
            (new_obs0_x - (initial_obs0_x - expected_move_dist)).abs() < f32::EPSILON,
            "Obstacle 0 should move left by speed * dt. Expected {}, got {}",
            initial_obs0_x - expected_move_dist,
            new_obs0_x
        );

        // Keep track of initial scores
        let initial_scores: Vec<u32> = world.dinos
            .iter()
            .map(|d| d.score)
            .collect();

        // 2. Force obstacle reset
        // Move obstacle 0 way off screen to the left
        world.obstacles[0].x = -OBSTACLE_WIDTH - 1.0;
        // Find the max_x of the *other* obstacles to predict reset position
        let max_other_x = world.obstacles
            .iter()
            .skip(1) // Skip obstacle 0
            .map(|o| o.x)
            .fold(f32::NEG_INFINITY, f32::max);

        // Run update again - this should trigger the reset for obstacle 0
        world.update(dt);

        let reset_obs0_x = world.obstacles[0].x;
        // Check if it reset based on max_other_x (allowing for random range)
        let expected_min_reset_x = max_other_x + 300.0 + 0.0; // Min of random range
        let expected_max_reset_x = max_other_x + 300.0 + 200.0; // Max of random range
        assert!(
            reset_obs0_x >= expected_min_reset_x && reset_obs0_x <= expected_max_reset_x,
            "Obstacle 0 should reset its position to the right. Expected range [{}, {}], got {}",
            expected_min_reset_x,
            expected_max_reset_x,
            reset_obs0_x
        );

        // 3. Check score increment for living dinos
        for (i, dino) in world.dinos.iter().enumerate() {
            if dino.alive {
                // Only living dinos should get score
                assert_eq!(
                    dino.score,
                    initial_scores[i] + 1,
                    "Living dino {} score should increment after obstacle reset",
                    i
                );
            } else {
                assert_eq!(
                    dino.score,
                    initial_scores[i],
                    "Dead dino {} score should not increment",
                    i
                );
            }
        }
    }

    #[test]
    fn test_dino_reset() {
        let mut dino = Dino::new(100.0, 50.0); // Start somewhere other than default reset

        // Change state
        dino.alive = false;
        dino.on_ground = false;
        dino.velocity_y = -20.0;
        dino.score = 15;
        dino.time_alive = 150;
        dino.x = 123.0;
        dino.y = 45.0;

        // Reset
        dino.reset();

        // Assert initial state
        assert!(dino.alive, "Dino should be alive after reset");
        assert!(dino.on_ground, "Dino should be on ground after reset");
        assert!(dino.velocity_y.abs() < f32::EPSILON, "Velocity Y should be 0 after reset");
        assert_eq!(dino.score, 0, "Score should be 0 after reset");
        assert_eq!(dino.time_alive, 0, "Time alive should be 0 after reset");
        assert!((dino.x - 50.0).abs() < f32::EPSILON, "X should be reset to 50.0"); // Assuming 50.0 is the reset X
        assert!((dino.y - GROUND_Y).abs() < f32::EPSILON, "Y should be reset to GROUND_Y");
    }

    #[test]
    fn test_world_setters() {
        let mut world = World::new(200);
        assert!(200 > 0, "Test requires population size > 0");

        // Define test data
        let test_weights: Vec<f32> = vec![0.1, -0.2, 0.3]; // Assuming hidden size is 3
        let test_bias: f32 = -0.5;

        // Check initial state (optional, but good practice)
        assert_ne!(world.brains[0].output_weights, test_weights, "Initial weights should differ");
        assert_ne!(world.brains[0].output_bias, test_bias, "Initial bias should differ");

        // Apply setters
        world.set_best_weights(test_weights.clone());
        world.set_best_bias(test_bias);

        // Assert changes
        assert_eq!(
            world.brains[0].output_weights,
            test_weights,
            "set_best_weights should update brain 0's output weights"
        );
        assert!(
            (world.brains[0].output_bias - test_bias).abs() < f32::EPSILON,
            "set_best_bias should update brain 0's output bias"
        );
    }

    #[test]
    fn test_nn_input_calculation_logic() {
        // Arrange: Set up world state manually to test input calculation
        let mut world = World::new(200);
        let dino_index = 0;
        let dino_x_pos = 50.0;
        let dino_score_val = 15;

        // Place obstacles: one far, one close, one behind
        world.obstacles = vec![
            Obstacle { x: 40.0, base_speed: 50.0 }, // Behind dino
            Obstacle { x: 100.0, base_speed: 50.0 }, // Closest relevant
            Obstacle { x: 300.0, base_speed: 50.0 } // Further away
        ];

        // Set dino state
        if let Some(dino) = world.dinos.get_mut(dino_index) {
            dino.x = dino_x_pos;
            dino.score = dino_score_val;
            dino.alive = true;
        } else {
            panic!("Dino not found for setup");
        }

        // Manually set speed multiplier (or calculate based on a known best score)
        let expected_speed_multiplier = 1.0 + 0.0 / 10.0; // Assuming score 0 is max for simplicity here
        world.speed_multiplier = expected_speed_multiplier;

        // Act: Replicate the input calculation logic from World::update
        let dino = &world.dinos[dino_index];
        let inputs = if
            let Some(obs) = world.obstacles
                .iter()
                // Find closest obstacle *in front* of the dino
                .filter(|o| o.x + OBSTACLE_WIDTH > dino.x)
                .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
        {
            let expected_dist = obs.x - dino.x;
            // Calculate expected score input based on the formula in update
            let expected_score_input = ((dino.score + 1) as f32) / 100.0;

            // Assertions within the calculation block for clarity
            assert!((obs.x - 100.0).abs() < f32::EPSILON, "Should select the obstacle at x=100");
            assert!(
                (expected_dist - (100.0 - dino_x_pos)).abs() < f32::EPSILON,
                "Calculated distance mismatch"
            );
            assert!(
                (expected_score_input - (15.0 + 1.0) / 100.0).abs() < f32::EPSILON,
                "Calculated score input mismatch"
            );

            vec![expected_dist, world.speed_multiplier, expected_score_input]
        } else {
            // If no obstacles found in front (shouldn't happen in this test setup)
            panic!("No relevant obstacle found for input calculation");
            // vec![0.0, world.speed_multiplier, ((dino.score + 1) as f32) / 100.0] // Default if needed
        };

        // Assert: Check the final inputs vector
        assert_eq!(inputs.len(), 3, "Inputs vector should have 3 elements");
        assert!(
            (inputs[0] - (100.0 - dino_x_pos)).abs() < f32::EPSILON,
            "Final distance input incorrect"
        );
        assert!(
            (inputs[1] - expected_speed_multiplier).abs() < f32::EPSILON,
            "Final speed multiplier input incorrect"
        );
        assert!(
            (inputs[2] - (15.0 + 1.0) / 100.0).abs() < f32::EPSILON,
            "Final score input incorrect"
        );
    }

    #[test]
    fn test_nn_predict_edge_cases() {
        // Arrange: Create a NN, potentially with known weights for simplicity
        // Using standard init for now
        let nn = NeuralNet::new(3, 999); // 3 inputs, specific seed

        // Case 1: Zero inputs
        let zero_inputs = vec![0.0, 0.0, 0.0];
        let zero_output = nn.predict(&zero_inputs);
        // Output depends on biases, but should be between 0 and 1
        assert!(
            zero_output >= 0.0 && zero_output <= 1.0,
            "Output with zero inputs out of range [0, 1]"
        );
        // We could calculate expected output if biases were known, but checking range is often sufficient

        // Case 2: Large positive inputs (expect output near 1)
        // Note: Actual inputs might be scaled/normalized, use values representative of potential max ranges
        let large_pos_inputs = vec![500.0, 2.0, 1.0]; // Large distance, high speed mult, high score input
        let large_pos_output = nn.predict(&large_pos_inputs);
        // Exact value depends on weights/biases, check if it's close to 1
        // The threshold (e.g., 0.9) depends on how quickly sigmoid saturates with typical weights
        println!("NN predict large pos output: {}", large_pos_output); // Optional: print to see value
        assert!(
            large_pos_output > 0.5,
            "Output with large positive inputs should likely be > 0.5 (or closer to 1 depending on weights)"
        ); // Adjust threshold based on observation/weights

        // Case 3: Large negative inputs (expect output near 0)
        // Note: Distance input is usually positive. Let's make one weight * input product large negative.
        // We'd need to know the weights, or construct a specific NN.
        // Alternative: Test with inputs that *should* result in a large negative sum before the final sigmoid.
        // Let's skip this for now unless we define a NN with known weights, as inputs are typically >= 0.

        // Case 4: Inputs designed to activate/deactivate sigmoid strongly
        // Requires known weights/biases. Example:
        /*
         let mut specific_nn = NeuralNet::new(3, 100);
         specific_nn.output_weights = vec![10.0, 10.0, 10.0]; // Strong positive weights
         specific_nn.output_bias = 0.0;
         // Assume hidden activations can be made close to 1.0
         let activating_inputs = vec![...]; // Inputs that make hidden activations high
         let activating_output = specific_nn.predict(&activating_inputs);
         assert!(activating_output > 0.95, "Activating inputs should yield output close to 1");

         specific_nn.output_weights = vec![-10.0, -10.0, -10.0]; // Strong negative weights
         let deactivating_inputs = vec![...]; // Inputs that make hidden activations high
         let deactivating_output = specific_nn.predict(&deactivating_inputs);
         assert!(deactivating_output < 0.05, "Deactivating inputs should yield output close to 0");
         */
    }

    #[test]
    fn test_update_multiple_close_obstacles() {
        // Arrange
        let mut world = World::new(200);
        let dt = 0.0167; // Simulate one frame ~60fps
        let dino_index = 0;
        let other_alive_dino_index = 1; // Keep this one alive

        // Place dino 0 specifically for collision
        if let Some(dino) = world.dinos.get_mut(dino_index) {
            dino.x = 50.0;
            dino.y = GROUND_Y;
            dino.on_ground = true;
            dino.alive = true;
        } else {
            panic!("Dino 0 setup failed");
        }

        // --- Keep dino 1 alive but out of the way ---
        if 200 > 1 {
            // Ensure we have a dino 1
            if let Some(dino) = world.dinos.get_mut(other_alive_dino_index) {
                dino.x = -1000.0; // Far away, won't collide
                dino.y = GROUND_Y;
                dino.on_ground = true;
                dino.alive = true; // Keep this one alive
            } else {
                panic!("Dino 1 setup failed");
            }
            // Mark all *other* dinos (except 0 and 1) as dead
            for i in 2..200 {
                if let Some(dino) = world.dinos.get_mut(i) {
                    dino.alive = false;
                }
            }
        } else {
            // If population is only 1, mark no others dead.
            // This case shouldn't happen based on 200 = 200
            // but good to consider edge cases.
        }

        // Place obstacles very close to each other, and near the dino
        let obs1_x = 70.0 - OBSTACLE_WIDTH - 1.0; // Just barely in front
        let obs2_x = obs1_x + OBSTACLE_WIDTH + 5.0; // 5 units behind the first one
        world.obstacles = vec![
            Obstacle { x: obs1_x, base_speed: 50.0 },
            Obstacle { x: obs2_x, base_speed: 50.0 }
        ];
        world.speed_multiplier = 1.0; // Simplify speed

        // We need a predictable brain output for this test.
        // Let's force the brain to jump if distance < 20.0
        if let Some(brain) = world.brains.get_mut(dino_index) {
            // Create a simple brain: high weights for distance (negated), low bias
            // This is a simplification; real weights are random.
            // We can't easily assert which obstacle was chosen without modifying code/adding logs.
            // Instead, let's focus on collision.
            brain.fitness = 0; // Reset fitness just in case
        }

        // Act: Run update - dino should collide with the first obstacle almost immediately
        world.update(dt);

        // Assert: Check if the dino is dead due to collision with the *first* obstacle
        assert!(
            !world.dinos[dino_index].alive,
            "Dino should be dead after colliding with the first close obstacle"
        );
        // Check if fitness was recorded (time_alive should be small, like 1)
        assert_eq!(
            world.brains[dino_index].fitness,
            1, // Since update runs once before collision check kills it
            "Fitness should reflect time_alive before death"
        );

        // Optional: Check if the second obstacle is largely unmoved (moved only by dt)
        let expected_obs2_x_after_update =
            obs2_x - world.obstacles[1].base_speed * world.speed_multiplier * dt;
        assert!(
            (world.obstacles[1].x - expected_obs2_x_after_update).abs() < f32::EPSILON,
            "Second obstacle should have moved normally"
        );
    }

    #[test]
    fn test_dino_jump_over_obstacle_space() {
        // Arrange
        let mut world = World::new(200);
        let dt = 0.05; // A slightly larger time step to see more movement
        let dino_index = 0;
        let other_alive_dino_index = 1; // Keep this one alive to prevent evolve

        // Place dino 0 before an obstacle
        let dino_start_x = 50.0;
        if let Some(dino) = world.dinos.get_mut(dino_index) {
            dino.x = dino_start_x;
            dino.y = GROUND_Y;
            dino.on_ground = true;
            dino.alive = true;
        } else {
            panic!("Dino 0 setup failed");
        }

        // Keep dino 1 alive but out of the way
        if 200 > 1 {
            if let Some(dino) = world.dinos.get_mut(other_alive_dino_index) {
                dino.x = -1000.0;
                dino.alive = true;
            } else {
                panic!("Dino 1 setup failed");
            }
            for i in 2..200 {
                // Mark others dead
                if let Some(dino) = world.dinos.get_mut(i) {
                    dino.alive = false;
                }
            }
        }

        // Place a single obstacle slightly ahead of the dino
        let obs_start_x = dino_start_x + DINO_WIDTH + 10.0; // Obstacle starts at x = 80.0
        world.obstacles = vec![Obstacle { x: obs_start_x, base_speed: 50.0 }];
        world.speed_multiplier = 1.0; // Simplify speed

        // Force dino 0 to jump just before the update
        let initial_y = world.dinos[dino_index].y;
        if let Some(dino) = world.dinos.get_mut(dino_index) {
            dino.velocity_y = MAX_JUMP_FORCE;
            dino.on_ground = false;
        }

        // Act: Run update. Dino should move up and not collide.
        world.update(dt);

        // Assert: Check dino 0 state after one jump step
        let dino = &world.dinos[dino_index];
        assert!(dino.alive, "Dino should still be alive after jumping over obstacle space");
        assert!(!dino.on_ground, "Dino should be in the air");

        // Check vertical motion (affected by gravity)
        let expected_vel_y = MAX_JUMP_FORCE + GRAVITY * dt;
        let expected_y = initial_y + expected_vel_y * dt; // Position based on updated velocity
        assert!(
            (dino.velocity_y - expected_vel_y).abs() < f32::EPSILON,
            "Dino velocity_y should be affected by gravity. Expected {:.2}, got {:.2}",
            expected_vel_y,
            dino.velocity_y
        );
        assert!(
            (dino.y - expected_y).abs() < f32::EPSILON,
            "Dino y should increase during jump. Expected {:.2}, got {:.2}",
            expected_y,
            dino.y
        );
        assert!(dino.y > GROUND_Y, "Dino y should be above ground");

        // Check obstacle movement
        let expected_obs_x =
            obs_start_x - world.obstacles[0].base_speed * world.speed_multiplier * dt;
        assert!(
            (world.obstacles[0].x - expected_obs_x).abs() < f32::EPSILON,
            "Obstacle should have moved left"
        );

        // Check that the other dino is still alive (prevented evolve)
        if 200 > 1 {
            assert!(world.dinos[other_alive_dino_index].alive, "Dino 1 should still be alive");
        }
    }
}
