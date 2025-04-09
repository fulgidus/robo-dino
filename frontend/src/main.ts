import init, { World } from './rust/rust_dino.js';
function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}
let world: World;
const canvas = document.getElementById('main') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const fitnessCanvas = document.getElementById('fitness') as HTMLCanvasElement;
const fitnessCtx = fitnessCanvas.getContext('2d')!;
let fitnessHistory: number[] = [];

const weightsCanvas = document.getElementById('weightsCanvas') as HTMLCanvasElement;
const weightsCtx = weightsCanvas.getContext('2d')!;

const nnCanvas = document.getElementById('neuralNet') as HTMLCanvasElement;
const nnCtx = nnCanvas.getContext('2d')!;

function weightToColor(weight: number): string {
    const normalized = (weight + 1) / 2; // da [-1, 1] → [0, 1]
    const r = Math.floor(normalized * 255);
    const b = Math.floor((1 - normalized) * 255);
    return `rgb(${r}, 0, ${b})`;
}
function drawWeightHeatmap() {
    const weights = world.get_best_input_weights(); // supponendo restituisca flat array
    const rows = 3; // numero fisso di righe
    const cols = Math.ceil(weights.length / rows); // numero di colonne calcolato
    const cellW = weightsCanvas.width / cols;  // larghezza della cella in base al numero di colonne
    const cellH = weightsCanvas.height / rows; // altezza della cella in base al numero di righe

    weightsCtx.clearRect(0, 0, weightsCanvas.width, weightsCanvas.height);

    for (let i = 0; i < weights.length; i++) {
        const row = Math.floor(i / cols); // calcola la riga in base al numero di colonne
        const col = i % cols; // calcola la colonna in base al numero di colonne
        weightsCtx.fillStyle = weightToColor(weights[i]);
        weightsCtx.fillRect(col * cellW, row * cellH, cellW, cellH); // disegna la cella
    }
}

function drawNeuron(x: number, y: number, r: number, color: string, label: string, value: number) {
    nnCtx.beginPath();
    nnCtx.arc(x, y, r, 0, 2 * Math.PI);
    nnCtx.fillStyle = color;
    nnCtx.fill();
    nnCtx.strokeStyle = 'black';
    nnCtx.stroke();
    nnCtx.fillStyle = 'black';
    nnCtx.font = '10px monospace';
    nnCtx.fillText(`${label}`, x - r, y - r - 2);
    nnCtx.fillText(value.toFixed(2), x - r + 3, y + 4);
}

function drawConnection(x1: number, y1: number, x2: number, y2: number, weight: number) {
    const normalized = Math.max(-1, Math.min(1, weight));
    const intensity = Math.floor(Math.abs(normalized) * 255);
    const color = normalized >= 0
        ? `rgb(${intensity},0,0)`  // red for positive
        : `rgb(${intensity / 2},${intensity / 2},${intensity})`; // blue for negative
    nnCtx.strokeStyle = color;
    nnCtx.lineWidth = 1 + Math.abs(normalized) * 2;
    nnCtx.beginPath();
    nnCtx.moveTo(x1, y1);
    nnCtx.lineTo(x2, y2);
    nnCtx.stroke();
}

function drawNeuralNet() {
    nnCtx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

    const inputLabels = ['Dist.', 'Vert. Speed', 'Horiz. Speed'];
    const weights = world.get_best_input_weights(); // flat [hidden][input]
    const outputWeights = world.get_best_output_weights();
    const outputBias = world.get_best_bias();

    const hiddenSize = outputWeights.length;
    const inputSize = inputLabels.length;
    const nodeRadius = 18;

    const inputPos: { x: number; y: number }[] = [];
    const hiddenPos: { x: number; y: number }[] = [];

    // 🔢 Calcola input reali del miglior dino
    const bestX = world.get_best_dino_x();
    const bestVY = world.get_best_dino_velocity?.() ?? 0;
    let minDist = Infinity;

    for (let i = 0; i < world.get_obstacle_count(); i++) {
        const ox = world.get_obstacle_x(i);
        const dx = ox - bestX;
        if (dx > 0 && dx < minDist) minDist = dx;
    }

    const normDist = minDist;
    const normVY = bestVY;
    const normScore = (world.get_best_score() + 1) / 100;
    const input = [normDist, normVY, normScore];

    // 🔧 Posizionamento neuroni
    function layerPosition(index: number, columnX: number, singleFile = false): [number, number] {
        const col = singleFile ? 1 : index % 2;
        const row = singleFile ? index + 0.5 : Math.floor(index / 2);
        const spacingX = 40;
        const spacingY = 60;
        return [columnX + col * spacingX, 50 + row * spacingY];
    }

    // ⬅️ Input neuron positions
    for (let i = 0; i < inputSize; i++) {
        const [x, y] = layerPosition(i, 50, true);
        inputPos.push({ x, y });
    }

    // 🔷 Hidden neuron positions
    for (let j = 0; j < hiddenSize; j++) {
        const [x, y] = layerPosition(j, 300);
        hiddenPos.push({ x, y });
    }

    // 🔴 Output neuron position
    const ox = 510;
    const oy = 120;

    // 🔌 Connessioni input → hidden
    for (let j = 0; j < hiddenSize; j++) {
        for (let i = 0; i < inputSize; i++) {
            const from = inputPos[i];
            const to = hiddenPos[j];
            const w = weights[j * inputSize + i];
            drawConnection(from.x, from.y, to.x, to.y, w);
        }
    }

    // 🔌 Connessioni hidden → output
    for (let j = 0; j < hiddenSize; j++) {
        const from = hiddenPos[j];
        drawConnection(from.x, from.y, ox, oy, outputWeights[j]);
    }

    // 🎯 Calcola attivazioni hidden
    const hiddenActivations: number[] = [];
    for (let j = 0; j < hiddenSize; j++) {
        const sum = input.reduce((acc, val, i) => acc + val * weights[j * inputSize + i], 0);
        const activation = 1 / (1 + Math.exp(-sum));
        hiddenActivations.push(activation);
    }

    // 🎯 Calcola output
    const dot = hiddenActivations.reduce((sum, h, i) => sum + h * outputWeights[i], 0);
    const outputSum = dot + outputBias;
    const outputActivation = sigmoid(outputSum);

    // 🧠 Disegna input neurons
    for (let i = 0; i < inputSize; i++) {
        const { x, y } = inputPos[i];
        drawNeuron(x, y, nodeRadius, 'lightblue', inputLabels[i], input[i]);
    }

    // 🧠 Disegna hidden neurons
    for (let j = 0; j < hiddenSize; j++) {
        const { x, y } = hiddenPos[j];
        drawNeuron(x, y, nodeRadius, 'teal', '', hiddenActivations[j]);
    }

    // 🧠 Disegna output neuron
    drawNeuron(ox, oy, nodeRadius, 'violet', 'Jump', outputActivation);
}



function draw() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Ground
    ctx.fillStyle = 'brown';
    ctx.fillRect(0, height - 20, width, 20);

    // Dino swarm
    const count = world.get_population_size?.() ?? 1;
    const dinoSize = 20;

    for (let i = 0; i < count; i++) {
        const x = world.get_dino_x(i);
        const y = world.get_dino_y(i);
        const alive = world.is_alive(i);
        const screenY = height - 20 - y - dinoSize;

        // Corpo dino
        ctx.fillStyle = alive ? 'green' : 'transparent';
        ctx.fillRect(x, screenY, dinoSize, dinoSize);

        // Bordo bianco
        ctx.strokeStyle = alive ? 'white' : 'transparent';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, screenY, dinoSize, dinoSize);
    }

    // Obstacles
    for (let i = 0; i < world.get_obstacle_count(); i++) {
        const ox = world.get_obstacle_x(i);
        ctx.fillStyle = 'red';
        ctx.fillRect(ox, height - 20 - 30, 5, 30);
    }

    // Info
    const score = world.get_best_score();
    // const avg = world.get_average_score().toFixed(2);
    const alive = world.count_alive();
    ctx.fillStyle = 'black';
    ctx.font = '14px monospace';
    ctx.fillText(`Score: ${score}`, 10, 20);
    ctx.fillText(`Alive: ${alive}`, 100, 20);
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
    if (!paused) {
        world.update(1 / 60);
    }

    draw();
    drawWeightHeatmap();
    drawFitnessGraph(fitnessHistory);
    drawNeuralNet();



    if (!paused) {
        const score = world.get_best_score();
        fitnessHistory.push(score);
        // if (fitnessHistory.length > 100) fitnessHistory.shift();
    }

    requestAnimationFrame(loop);
}

init().then(() => {
    world = new World();
    requestAnimationFrame(loop);
});

let paused = false;
document.getElementById('togglePause')!.addEventListener('click', () => {
    paused = !paused;
    const btn = document.getElementById('togglePause')!;
    btn.textContent = paused ? '▶️ Resume' : '⏸️ Pause';
});