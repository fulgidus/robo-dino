let wasm;

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let WASM_VECTOR_LEN = 0;

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

const ObstacleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_obstacle_free(ptr >>> 0, 1));

export class Obstacle {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ObstacleFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_obstacle_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.__wbg_get_obstacle_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set x(arg0) {
        wasm.__wbg_set_obstacle_x(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get base_speed() {
        const ret = wasm.__wbg_get_obstacle_base_speed(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set base_speed(arg0) {
        wasm.__wbg_set_obstacle_base_speed(this.__wbg_ptr, arg0);
    }
}

const WorldFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_world_free(ptr >>> 0, 1));

export class World {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WorldFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_world_free(ptr, 0);
    }
    /**
     * @param {number} count
     */
    constructor(count) {
        const ret = wasm.world_new(count);
        this.__wbg_ptr = ret >>> 0;
        WorldFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} dt
     */
    update(dt) {
        wasm.world_update(this.__wbg_ptr, dt);
    }
    /**
     * @returns {number}
     */
    get_best_dino_x() {
        const ret = wasm.world_get_best_dino_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get_best_dino_y() {
        const ret = wasm.world_get_best_dino_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get_best_dino_velocity_y() {
        const ret = wasm.world_get_best_dino_velocity_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get_best_index() {
        const ret = wasm.world_get_best_index(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_best_score() {
        const ret = wasm.world_get_best_score(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_generation() {
        const ret = wasm.world_get_generation(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_obstacle_count() {
        const ret = wasm.world_get_obstacle_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} index
     * @returns {number}
     */
    get_obstacle_x(index) {
        const ret = wasm.world_get_obstacle_x(this.__wbg_ptr, index);
        return ret;
    }
    /**
     * @returns {Uint32Array}
     */
    get_fitness_history() {
        const ret = wasm.world_get_fitness_history(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {number} index
     * @returns {number}
     */
    get_score_of(index) {
        const ret = wasm.world_get_score_of(this.__wbg_ptr, index);
        return ret >>> 0;
    }
    /**
     * @returns {Float32Array}
     */
    get_best_input_weights() {
        const ret = wasm.world_get_best_input_weights(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float32Array}
     */
    get_best_output_weights() {
        const ret = wasm.world_get_best_output_weights(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get_best_bias() {
        const ret = wasm.world_get_best_bias(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Float32Array} weights
     */
    set_best_weights(weights) {
        const ptr0 = passArrayF32ToWasm0(weights, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.world_set_best_weights(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} bias
     */
    set_best_bias(bias) {
        wasm.world_set_best_bias(this.__wbg_ptr, bias);
    }
    /**
     * @returns {number}
     */
    count_alive() {
        const ret = wasm.world_count_alive(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_average_score() {
        const ret = wasm.world_get_average_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} index
     * @returns {boolean}
     */
    is_alive(index) {
        const ret = wasm.world_is_alive(this.__wbg_ptr, index);
        return ret !== 0;
    }
    /**
     * @returns {Float32Array}
     */
    get_best_hidden_biases() {
        const ret = wasm.world_get_best_hidden_biases(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get_population_size() {
        const ret = wasm.world_get_population_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} index
     * @returns {number}
     */
    get_dino_x(index) {
        const ret = wasm.world_get_dino_x(this.__wbg_ptr, index);
        return ret;
    }
    /**
     * @param {number} index
     * @returns {number}
     */
    get_dino_y(index) {
        const ret = wasm.world_get_dino_y(this.__wbg_ptr, index);
        return ret;
    }
    /**
     * @returns {number}
     */
    get_speed_multiplier() {
        const ret = wasm.world_get_speed_multiplier(this.__wbg_ptr);
        return ret;
    }
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                if (module.headers.get('Content-Type') != 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_error_524f506f44df1645 = function(arg0) {
        console.error(arg0);
    };
    imports.wbg.__wbg_log_c222819a41e063d3 = function(arg0) {
        console.log(arg0);
    };
    imports.wbg.__wbg_warn_4ca3906c248c47c4 = function(arg0) {
        console.warn(arg0);
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_0;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };
    imports.wbg.__wbindgen_string_new = function(arg0, arg1) {
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('rust_dino_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
