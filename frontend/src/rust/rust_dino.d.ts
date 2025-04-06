/* tslint:disable */
/* eslint-disable */
export class Obstacle {
  private constructor();
  free(): void;
  x: number;
  base_speed: number;
}
export class World {
  free(): void;
  constructor();
  update(dt: number): void;
  get_best_dino_x(): number;
  get_best_dino_y(): number;
  get_best_score(): number;
  get_generation(): number;
  get_obstacle_count(): number;
  get_obstacle_x(index: number): number;
  get_fitness_history(): Uint32Array;
  get_score(): number;
  get_best_weights(): Float32Array;
  get_best_bias(): number;
  set_best_weights(weights: Float32Array): void;
  set_best_bias(bias: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_obstacle_free: (a: number, b: number) => void;
  readonly __wbg_get_obstacle_x: (a: number) => number;
  readonly __wbg_set_obstacle_x: (a: number, b: number) => void;
  readonly __wbg_get_obstacle_base_speed: (a: number) => number;
  readonly __wbg_set_obstacle_base_speed: (a: number, b: number) => void;
  readonly __wbg_world_free: (a: number, b: number) => void;
  readonly world_new: () => number;
  readonly world_update: (a: number, b: number) => void;
  readonly world_get_best_dino_x: (a: number) => number;
  readonly world_get_best_dino_y: (a: number) => number;
  readonly world_get_best_score: (a: number) => number;
  readonly world_get_generation: (a: number) => number;
  readonly world_get_obstacle_count: (a: number) => number;
  readonly world_get_obstacle_x: (a: number, b: number) => number;
  readonly world_get_fitness_history: (a: number) => [number, number];
  readonly world_get_score: (a: number) => number;
  readonly world_get_best_weights: (a: number) => [number, number];
  readonly world_get_best_bias: (a: number) => number;
  readonly world_set_best_weights: (a: number, b: number, c: number) => void;
  readonly world_set_best_bias: (a: number, b: number) => void;
  readonly __wbindgen_export_0: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
