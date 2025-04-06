import init, { World } from './rust/rust_dino.js';

const NUM_INSTANCES = 16;
const instances: Instance[] = [];

type Instance = {
    world: World;
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
};

const fitnessCanvas = document.getElementById('fitness') as HTMLCanvasElement;
const fitnessCtx = fitnessCanvas.getContext('2d')!;
let fitnessHistory: number[] = [];

function saveBestBrain(world: World) {
    const weights = world.get_best_weights();
    const bias = world.get_best_bias();
    const brain = { weights, bias };
    localStorage.setItem('best_brain', JSON.stringify(brain));
}

function loadBestBrain(): { weights: number[]; bias: number } | null {
    const raw = localStorage.getItem('best_brain');
    if (raw) return JSON.parse(raw);
    return null;
}

function createCanvasGrid() {
    const container = document.getElementById('grid')!;
    for (let i = 0; i < NUM_INSTANCES; i++) {
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 100;
        container.appendChild(canvas);

        instances.push({
            canvas,
            ctx: canvas.getContext('2d')!,
            world: null as any,
        });
    }
}

function drawInstance(instance: Instance) {
    const { ctx, world, canvas } = instance;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const groundHeight = 20;
    const dinoW = 20, dinoH = 20, obsW = 20, obsH = 30;

    ctx.fillStyle = 'brown';
    ctx.fillRect(0, canvas.height - groundHeight, canvas.width, groundHeight);

    const dx = world.get_best_dino_x();
    const dy = world.get_best_dino_y();
    const dinoY = canvas.height - groundHeight - dy - dinoH;

    ctx.fillStyle = 'green';
    ctx.fillRect(dx, dinoY, dinoW, dinoH);

    for (let i = 0; i < world.get_obstacle_count(); i++) {
        const ox = world.get_obstacle_x(i);
        const oy = canvas.height - groundHeight - obsH;
        ctx.fillStyle = 'red';
        ctx.fillRect(ox, oy, obsW, obsH);
    }

    ctx.fillStyle = 'black';
    ctx.font = '12px monospace';
    ctx.fillText(`Score: ${world.get_score()}`, 5, 12);
}

function drawFitnessGraph(history: number[]) {
    fitnessCtx.clearRect(0, 0, fitnessCanvas.width, fitnessCanvas.height);
    const max = Math.max(...history, 10);
    const width = fitnessCanvas.width;
    const height = fitnessCanvas.height;

    fitnessCtx.beginPath();
    fitnessCtx.moveTo(0, height);

    history.forEach((v, i) => {
        const x = (i / history.length) * width;
        const y = height - (v / max) * height;
        fitnessCtx.lineTo(x, y);
    });

    fitnessCtx.strokeStyle = 'blue';
    fitnessCtx.lineWidth = 2;
    fitnessCtx.stroke();

    fitnessCtx.strokeStyle = 'gray';
    const bestY = height - (Math.max(...history) / max) * height;
    fitnessCtx.beginPath();
    fitnessCtx.moveTo(0, bestY);
    fitnessCtx.lineTo(width, bestY);
    fitnessCtx.stroke();
}

function loop() {
    const dt = 1 / 60;
    let bestScore = 0;
    let bestWorld: World | null = null;

    for (const inst of instances) {
        inst.world.update(dt);
        drawInstance(inst);
        const score = inst.world.get_score();
        if (score > bestScore) {
            bestScore = score;
            bestWorld = inst.world;
        }
    }

    if (bestWorld) {
        saveBestBrain(bestWorld);
    }

    fitnessHistory.push(bestScore);
    if (fitnessHistory.length > 100) fitnessHistory.shift();
    drawFitnessGraph(fitnessHistory);

    requestAnimationFrame(loop);
}

init().then(() => {
    createCanvasGrid();
    for (const inst of instances) {
        inst.world = new World();

        // carica cervello salvato nel primo dino solo
        const saved = loadBestBrain();
        if (saved) {
            inst.world.set_best_weights(Float32Array.from(saved.weights));
            inst.world.set_best_bias(saved.bias);
        }
    }

    // pulsanti
    document.getElementById('reset-learning')!.addEventListener('click', () => {
        localStorage.removeItem('best_brain');
        location.reload();
    });

    document.getElementById('clear-data')!.addEventListener('click', () => {
        localStorage.removeItem('best_brain');
        alert('Saved data cleared!');
    });

    requestAnimationFrame(loop);
});
