import init, { World } from './rust/rust_dino.js';

let world: World;
const canvas = document.getElementById('main') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const fitnessCanvas = document.getElementById('fitness') as HTMLCanvasElement;
const fitnessCtx = fitnessCanvas.getContext('2d')!;
let fitnessHistory: number[] = [];

const weightsCanvas = document.getElementById('weightsCanvas') as HTMLCanvasElement;
const weightsCtx = weightsCanvas.getContext('2d')!;

function weightToColor(weight: number): string {
    const normalized = (weight + 1) / 2; // da [-1, 1] → [0, 1]
    const r = Math.floor(normalized * 255);
    const b = Math.floor((1 - normalized) * 255);
    return `rgb(${r}, 0, ${b})`;
}

function drawWeightHeatmap() {
    const weights = world.get_best_input_weights(); // supponendo restituisca flat array
    const cols = 3; // numero input
    const rows = weights.length / cols;
    const cellW = weightsCanvas.width / cols;
    const cellH = weightsCanvas.height / rows;

    weightsCtx.clearRect(0, 0, weightsCanvas.width, weightsCanvas.height);

    for (let i = 0; i < weights.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        weightsCtx.fillStyle = weightToColor(weights[i]);
        weightsCtx.fillRect(col * cellW, row * cellH, cellW, cellH);
    }
}

function draw() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Ground
    ctx.fillStyle = 'brown';
    ctx.fillRect(0, height - 20, width, 20);

    // dino (all)
    const count = world.get_population_size?.() ?? 1;
    const size = 20;
    const border = 2;

    for (let i = 0; i < count; i++) {
        const alive = world.is_alive(i);
        const dx = world.get_dino_x(i);
        const dy = world.get_dino_y(i);
        const size = 20;
        const screenY = height - 20 - dy - size;

        // Corpo
        ctx.fillStyle = alive ? 'green' : 'transparent';
        ctx.fillRect(dx, screenY, size, size);

        // Bordo visibile solo se vivo
        if (alive) {
            ctx.save(); // salva stato
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'white';
            ctx.strokeRect(dx, screenY, size, size);
            ctx.restore(); // ripristina
        }
    }
    // Obstacles
    for (let i = 0; i < world.get_obstacle_count(); i++) {
        const ox = world.get_obstacle_x(i);
        ctx.fillStyle = 'red';
        ctx.fillRect(ox, height - 20 - 30, 20, 30);
    }

    // Info
    let bestScore = 0;
    for (let i = 0; i < count; i++) {
        if (world.is_alive(i)) {
            bestScore = Math.max(bestScore, world.get_score_of(i));
        }
    } const avg = world.get_average_score().toFixed(2);
    const alive = world.count_alive();
    ctx.fillStyle = 'black';
    ctx.font = '14px monospace';
    ctx.fillText(`Score: ${bestScore}`, 10, 20);
    ctx.fillText(`Alive: ${alive}`, 150, 20);
}

function drawFitnessGraph(history: number[]) {
    fitnessCtx.clearRect(0, 0, fitnessCanvas.width, fitnessCanvas.height);
    const max = Math.max(...history, 10);
    const w = fitnessCanvas.width;
    const h = fitnessCanvas.height;

    fitnessCtx.beginPath();
    fitnessCtx.moveTo(0, h);
    history.forEach((v, i) => {
        const x = (i / history.length) * w;
        const y = h - (v / max) * h;
        fitnessCtx.lineTo(x, y);
    });
    fitnessCtx.strokeStyle = 'blue';
    fitnessCtx.stroke();
}

function loop() {
    world.update(1 / 60);
    draw();
    drawWeightHeatmap();
    const score = world.get_best_score();
    fitnessHistory.push(score);
    if (fitnessHistory.length > 100) fitnessHistory.shift();
    drawFitnessGraph(fitnessHistory);
    requestAnimationFrame(loop);


    const alive = world.count_alive();
    console.log("Alive (frontend):", alive);
}

init().then(() => {
    world = new World();
    requestAnimationFrame(loop);
});