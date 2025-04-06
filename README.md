# 🦖 Robo Dino – Rust + WebAssembly + TypeScript + Canvas

A minimalist clone of the Chrome Dino Runner, with the game logic written in **Rust**, compiled to **WebAssembly**, and rendered using **Vanilla TypeScript + Canvas** via **Vite**.

## 📦 Requirements

- [Rust](https://www.rust-lang.org/tools/install)
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/)  
  Install via:
  ```bash
  cargo install wasm-pack
	```
- [Node.js + npm](https://nodejs.org) or [pnpm](https://pnpm.io)
   
## 🚀 Project structure

```bash
robo-dino/
├── rust-dino/             # Rust code (compiled to WASM)
│   └── src/lib.rs         # game logic
├── frontend/              # frontend using Vite + TypeScript + Canvas
│   ├── index.html
│   ├── main.ts
│   └── src/rust/          # output of wasm-pack build (JS + .wasm)
```

## 🛠️ Building the WASM module

From the root or `rust-dino/` folder, run:

```bash
wasm-pack build --target web --out-dir ../frontend/src/rust
```

This will:

- Compile the Rust code to WebAssembly
- Output JavaScript + `.wasm` bindings to `frontend/src/rust/`
- The module is then imported in `main.ts` like so:
	```ts
  import init, { World } from './rust/rust_dino.js';
	  ```
## 🎮 Controls

- **Click** on the canvas to make the dino jump
- Obstacles get faster as your score increases
- Collisions reset your score (you lose!)
## ✨ Tech Stack

- 🦀 **Rust** – high-performance game logic
- ⚙️ **wasm-pack** – compiles Rust to WASM + JS bindings
- 🧠 **TypeScript** – type-safe frontend
- 🎨 **Canvas 2D API** – lightweight rendering
- ⚡ **Vite** – ultra-fast dev server + build tool

## 🤝 Credits

Built as a fun technical experiment to explore WebAssembly and Rust in a modern frontend setup.  
A "labour-of-love" side project made to learn, tinker, and watch a dino run across your browser.


## 📸 Screenshot

_(Coming soon)_