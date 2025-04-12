import init, { World } from './rust/rust_dino.js';

let world: World;
let speedMultiplier = 1;
let agentCount = 200;
let showAll = true;
let showOnlyAlive = true;


const canvas = document.getElementById('main') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const fitnessCanvas = document.getElementById('fitness') as HTMLCanvasElement;
const fitnessCtx = fitnessCanvas.getContext('2d')!;
let fitnessHistory: number[] = [];

const weightsCanvas = document.getElementById('weightsCanvas') as HTMLCanvasElement;
const weightsCtx = weightsCanvas.getContext('2d')!;

const nnCanvas = document.getElementById('neuralNet') as HTMLCanvasElement;
const nnCtx = nnCanvas.getContext('2d')!;

// Definisci colori base per i neuroni
const inputColor = 'rgb(173, 216, 230)'; // lightblue
const hiddenColor = 'rgb(0, 128, 128)';   // teal
const outputColor = 'rgb(238, 130, 238)'; // violet

// --- Funzioni Helper ---

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

function interpolateColor(color1: string, color2: string, factor: number): string {
    let r1 = 0, g1 = 0, b1 = 0, r2 = 0, g2 = 0, b2 = 0;
    try {
        // Estrai componenti RGB dal primo colore
        const match1 = color1.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (match1) { [r1, g1, b1] = match1.slice(1).map(Number); }
        else { return color1; } // Fallback se il formato non è corretto

        // Imposta componenti RGB per il secondo colore (assumiamo bianco o nero per semplicità)
        if (color2 === 'white') { [r2, g2, b2] = [255, 255, 255]; }
        else if (color2 === 'black') { [r2, g2, b2] = [0, 0, 0]; }
        else { // Prova a parsare anche il secondo colore se non è bianco/nero
            const match2 = color2.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
            if (match2) { [r2, g2, b2] = match2.slice(1).map(Number); }
            else return color1; // Fallback
        }
    } catch (e) {
        console.error("Color parsing/interpolation failed", e);
        return color1; // Fallback in caso di errore
    }

    // Calcola il colore interpolato
    const r = Math.round(r1 + (r2 - r1) * factor);
    const g = Math.round(g1 + (g2 - g1) * factor);
    const b = Math.round(b1 + (b2 - b1) * factor);
    return `rgb(${r}, ${g}, ${b})`;
}

function weightToColor(weight: number): string {
    const normalized = (weight + 1) / 2; // da [-1, 1] → [0, 1]
    const r = Math.floor(normalized * 255);
    const b = Math.floor((1 - normalized) * 255);
    return `rgb(${r}, 0, ${b})`;
}
function drawWeightHeatmap() {
    const weights = world.get_best_input_weights();
    const rows = 3;
    const cols = Math.ceil(weights.length / rows);
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

function drawNeuron(
    ctx: CanvasRenderingContext2D, // Passa il contesto esplicitamente
    x: number,
    y: number,
    baseRadius: number,
    baseColor: string,
    label: string,
    displayValue: number,
    activationValue: number, // Valore di attivazione per gli effetti
    time: number // Tempo per l'animazione di pulsazione
) {
    const clampedActivation = Math.max(0, Math.min(1, activationValue)); // Assicura 0-1

    // --- Effetto Pulsazione ---
    const pulseFrequency = 0.005; // Velocità della pulsazione
    const pulseAmplitude = baseRadius * 0.2; // Ampiezza massima (20% del raggio)
    // L'ampiezza effettiva dipende dall'attivazione
    const currentPulseAmplitude = pulseAmplitude * clampedActivation;
    const radius = baseRadius + currentPulseAmplitude * Math.sin(pulseFrequency * time);

    // --- Effetto Luminosità ---
    // Interpola verso il bianco in base all'attivazione (es. 30% a max attivazione)
    const highlightFactor = clampedActivation * 0.3;
    const finalColor = interpolateColor(baseColor, 'white', highlightFactor);

    // Disegna il neurone
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI); // Usa raggio pulsante
    ctx.fillStyle = finalColor; // Usa colore con luminosità variabile
    ctx.fill();
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1; // Resetta lo spessore linea se modificato altrove
    ctx.stroke();

    // Disegna testo (etichetta e valore)
    ctx.fillStyle = 'black';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    // Posiziona l'etichetta sopra, tenendo conto del raggio variabile
    ctx.fillText(`${label}`, x, y - radius - 5);
    // Posiziona il valore all'interno
    ctx.fillText(displayValue.toFixed(2), x, y + 4);
    ctx.textAlign = 'start'; // Resetta l'allineamento
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

function drawNnLegend(ctx: CanvasRenderingContext2D, startX: number, startY: number) {
    const legendFont = '10px monospace';
    const lineHeight = 12;
    let currentY = startY;

    ctx.fillStyle = 'black';
    ctx.font = legendFont;
    ctx.fillText("Legend:", startX, currentY);
    currentY += lineHeight * 1.5;

    // Legenda Connessioni
    ctx.fillText("Connections:", startX, currentY);
    currentY += lineHeight;
    // Positiva (Rossa, Spessa)
    ctx.strokeStyle = 'rgb(255,0,0)';
    ctx.lineWidth = 1 + 1 * 2; // Esempio spessore massimo
    ctx.beginPath(); ctx.moveTo(startX, currentY); ctx.lineTo(startX + 20, currentY); ctx.stroke();
    ctx.fillStyle = 'black'; ctx.fillText("+ Weight (Intensity)", startX + 25, currentY + 3);
    currentY += lineHeight;
    // Negativa (Blu, Spessa)
    ctx.strokeStyle = `rgb(127,127,255)`; // Esempio blu intenso
    ctx.lineWidth = 1 + 1 * 2; // Esempio spessore massimo
    ctx.beginPath(); ctx.moveTo(startX, currentY); ctx.lineTo(startX + 20, currentY); ctx.stroke();
    ctx.fillStyle = 'black'; ctx.fillText("- Weight (Intensity)", startX + 25, currentY + 3);
    currentY += lineHeight * 1.5;

    // Legenda Neuroni
    ctx.fillText("Neurons:", startX, currentY);
    currentY += lineHeight;
    // Input
    ctx.fillStyle = inputColor;
    ctx.beginPath(); ctx.arc(startX + 10, currentY, 5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = 'black'; ctx.fillText("Input", startX + 25, currentY + 3);
    currentY += lineHeight;
    // Hidden
    ctx.fillStyle = hiddenColor;
    ctx.beginPath(); ctx.arc(startX + 10, currentY, 5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = 'black'; ctx.fillText("Hidden", startX + 25, currentY + 3);
    currentY += lineHeight;
    // Output
    ctx.fillStyle = outputColor;
    ctx.beginPath(); ctx.arc(startX + 10, currentY, 5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = 'black'; ctx.fillText("Output (Jump)", startX + 25, currentY + 3);
    currentY += lineHeight * 1.5;

    // Legenda Attivazione
    ctx.fillText("Neuron activation:", startX, currentY);
    currentY += lineHeight;
    ctx.fillText("Pulsing/luminosity", startX + 5, currentY);
    currentY += lineHeight;
    ctx.fillText("= Neuron output", startX + 5, currentY);

    // Reset stili
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
}

function drawNeuralNet(time: number) {
    nnCtx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

    // --- Ottieni dati dal miglior dinosauro (usando l'indice corretto) ---
    const bestIndex = world.get_best_index();
    // Verifica se l'indice è valido prima di procedere
    if (typeof bestIndex !== 'number' || bestIndex < 0 || bestIndex >= world.get_population_size()) {
        console.warn("Invalid bestIndex received:", bestIndex);
        // Potresti disegnare un messaggio di errore o semplicemente non disegnare la rete
        nnCtx.fillStyle = 'red';
        nnCtx.font = '12px monospace';
        nnCtx.fillText("Waiting for best dino data...", 10, 20);
        return;
    }

    const weights = world.get_best_input_weights();
    const outputWeights = world.get_best_output_weights();
    const outputBias = world.get_best_bias();
    const hiddenBiases = world.get_best_hidden_biases();

    const hiddenSize = outputWeights.length;
    // Definisci le etichette e la dimensione input qui, basandoti sugli input REALI usati in Rust
    const inputLabels = ['Dist.', 'Speed Mult.', 'Score Norm.']; // Etichette corrispondenti agli input Rust
    const inputSize = inputLabels.length; // Dovrebbe essere 3
    const nodeRadius = 18;

    const inputPos: { x: number; y: number }[] = [];
    const hiddenPos: { x: number; y: number }[] = [];

    // --- CALCOLO INPUT CORRETTO (come in Rust) ---
    const bestX = world.get_dino_x(bestIndex);
    const bestScore = world.get_score_of(bestIndex); // Assicurati di avere questa funzione in Rust
    const currentSpeedMultiplier = world.get_speed_multiplier();

    // Trova la distanza dall'ostacolo più vicino *davanti* al best dino
    let minDist = Infinity;
    const OBSTACLE_WIDTH_JS = 5.0; // Assicurati sia uguale a OBSTACLE_WIDTH in Rust
    for (let i = 0; i < world.get_obstacle_count(); i++) {
        const ox = world.get_obstacle_x(i);
        // Logica per trovare l'ostacolo rilevante (come in Rust: o.x + OBSTACLE_WIDTH > dino.x)
        if (ox + OBSTACLE_WIDTH_JS > bestX) {
            const dx = ox - bestX; // Distanza dal bordo sinistro dell'ostacolo
            // Troviamo il minimo tra le distanze (anche negative se sovrapposto)
            if (dx < minDist) {
                minDist = dx;
            }
        }
    }

    // Se non ci sono ostacoli rilevanti (molto lontano o all'inizio), usa un valore grande.
    // Potresti voler affinare questo comportamento per matchare Rust perfettamente.
    const inputDistance = minDist === Infinity ? 1000.0 : minDist;

    // Crea il vettore input ESATTAMENTE come in Rust
    const input = [
        inputDistance,                      // 1. Distanza grezza
        currentSpeedMultiplier,             // 2. Moltiplicatore velocità grezzo
        (bestScore + 1) / 100.0             // 3. Punteggio normalizzato
    ];
    // --- FINE CALCOLO INPUT CORRETTO ---


    // --- Posizionamento neuroni (Layout) ---
    function layerPosition(index: number, columnX: number, count: number): [number, number] {
        // Centra verticalmente il layer
        const totalHeight = (count - 1) * 60; // Spaziatura verticale
        const startY = (nnCanvas.height - totalHeight) / 2;
        const y = startY + index * 60;
        return [columnX, y];
    }

    // Input neuron positions
    for (let i = 0; i < inputSize; i++) {
        const [x, y] = layerPosition(i, 50, inputSize);
        inputPos.push({ x, y });
    }
    // Hidden neuron positions
    for (let j = 0; j < hiddenSize; j++) {
        const [x, y] = layerPosition(j, 180, hiddenSize); // Spostato un po' a destra
        hiddenPos.push({ x, y });
    }
    // Output neuron position (centrato verticalmente)
    const [ox, oy] = layerPosition(0, 310, 1); // Spostato un po' a destra


    // --- Disegno Connessioni ---
    // Connessioni input → hidden
    for (let j = 0; j < hiddenSize; j++) {
        for (let i = 0; i < inputSize; i++) {
            const from = inputPos[i];
            const to = hiddenPos[j];
            // Accedi al peso corretto: weights[indice_neurone_hidden * num_input + indice_input]
            const w = weights[j * inputSize + i];
            drawConnection(from.x, from.y, to.x, to.y, w);
        }
    }
    // Connessioni hidden → output
    for (let j = 0; j < hiddenSize; j++) {
        const from = hiddenPos[j];
        drawConnection(from.x, from.y, ox, oy, outputWeights[j]);
    }


    // --- Calcola Attivazioni (Simulazione Forward Pass in TS) ---
    const hiddenActivations: number[] = [];
    for (let j = 0; j < hiddenSize; j++) {
        // Somma pesata degli input
        let sum = input.reduce((acc, val, i) => acc + val * weights[j * inputSize + i], 0);

        // !!! Applica il bias nascosto se esiste e viene usato in Rust !!!
        sum += hiddenBiases[j]; // Decommenta se necessario

        // Applica la funzione di attivazione (Sigmoid)
        const activation = sigmoid(sum);
        hiddenActivations.push(activation);
    }
    // Calcola output
    const dot = hiddenActivations.reduce((sum, h, i) => sum + h * outputWeights[i], 0);
    const outputSum = dot + outputBias; // Applica il bias di output
    const outputActivation = sigmoid(outputSum); // Applica Sigmoid


    // --- Disegna Neuroni ---
    // Input neurons
    for (let i = 0; i < inputSize; i++) {
        const { x, y } = inputPos[i];
        // L'attivazione visiva di un input è il suo valore stesso (o una sua normalizzazione se preferisci)
        // Usiamo il valore grezzo per il display, e lo stesso per l'effetto (clampato 0-1)
        const effectValue = Math.max(0, Math.min(1, input[i])); // Clamp per effetto visivo
        drawNeuron(nnCtx, x, y, nodeRadius, inputColor, inputLabels[i], input[i], effectValue, time);
    }
    // Hidden neurons
    for (let j = 0; j < hiddenSize; j++) {
        const { x, y } = hiddenPos[j];
        // L'attivazione è l'output del neurone nascosto
        drawNeuron(nnCtx, x, y, nodeRadius, hiddenColor, '', hiddenActivations[j], hiddenActivations[j], time);
    }
    // Output neuron
    const effectActivation = outputActivation > 0.6 ? outputActivation : 0; // Effetto condizionale
    // Passa outputActivation per il display, effectActivation per l'effetto
    drawNeuron(nnCtx, ox, oy, nodeRadius, outputColor, 'Jump', outputActivation, effectActivation, time);


    // --- Disegna la Legenda ---
    drawNnLegend(nnCtx, nnCanvas.width - 160, 20); // Posiziona in alto a destra
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
    const bestIndex = world.get_best_index?.(); // Ottieni l'indice una sola volta
    const hasValidBestIndex = typeof bestIndex === 'number' && bestIndex >= 0;

    for (let i = 0; i < count; i++) {
        const alive = world.is_alive(i);
        let skip = false;
        if (!showAll) {
            // Se mostriamo solo il migliore:
            // Salta se non abbiamo un indice valido O se l'indice corrente non è il migliore.
            alive && console.log('hasValidBestIndex', hasValidBestIndex, 'i', i, 'bestIndex', bestIndex, 'alive', alive, 'showAll', showAll);
            if (!hasValidBestIndex || i !== bestIndex) {
                skip = true;
            }
        }
        if (!skip && showOnlyAlive && !alive) {
            // Se non stavamo già saltando, E mostriamo solo i vivi, E questo è morto: salta.
            skip = true;
        }
        if (skip) {
            continue; // Passa al prossimo dinosauro
        }

        const x = world.get_dino_x(i);
        const y = world.get_dino_y(i);
        const screenY = height - 20 - y - dinoSize;
        const deadColor = 'rgba(128, 128, 128, 0.05)'; // Grigio con 10% opacità

        // Corpo dino
        ctx.fillStyle = alive ? 'green' : deadColor;
        ctx.fillRect(x, screenY, dinoSize, dinoSize);

        // Bordo bianco
        ctx.strokeStyle = alive ? 'white' : deadColor;
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
    const generation = world.get_generation();
    ctx.fillStyle = 'black';
    ctx.font = '14px monospace';
    ctx.fillText(`Score: ${score}`, 10, 20);
    ctx.fillText(`Alive: ${alive}`, 100, 20);
    ctx.fillText(`Generation: ${generation}`, 350, 20);
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

function loop(time: number, buff: any) {
    if (!paused) {
        world.update((1 / 60) * speedMultiplier);
    }

    /*     const ptr = world.export_velocity_ptr();
        const velocity = new Float32Array(buff, ptr, 1)[0];
    
        const velocityDataView = new DataView(buff).getInt32(ptr, true);
        console.log('velocityDataView', velocity, ptr, velocityDataView, buff)
     */
    draw();
    drawWeightHeatmap();
    drawFitnessGraph(fitnessHistory);
    drawNeuralNet(time);



    if (!paused) {
        const score = world.get_best_score();
        fitnessHistory.push(score);
        // if (fitnessHistory.length > 100) fitnessHistory.shift();
    }

    requestAnimationFrame((newTime) => loop(newTime, buff));
}


init().then((val) => {
    world = new World(agentCount);
    requestAnimationFrame((initialTime) => loop(initialTime, val.memory.buffer));
});

// --- Gestori Eventi ---
let paused = false;
// ... (event listener invariati per pausa, velocità, checkbox, apply settings) ...
document.getElementById('togglePause')!.addEventListener('click', () => {
    paused = !paused;
    const btn = document.getElementById('togglePause')!;
    btn.textContent = paused ? '▶️ Resume' : '⏸️ Pause';
});
document.getElementById('speed')!.addEventListener('input', (e) => {
    speedMultiplier = parseFloat((e.target as HTMLInputElement).value);
});
document.getElementById('showAll')!.addEventListener('change', (e) => {
    showAll = (e.target as HTMLInputElement).checked;
});
document.getElementById('showOnlyAlive')!.addEventListener('change', (e) => {
    showOnlyAlive = (e.target as HTMLInputElement).checked;
});
document.getElementById('applySettings')!.addEventListener('click', () => {
    const newCountInput = document.getElementById('agents') as HTMLInputElement;
    const newCount = parseInt(newCountInput.value);
    if (!isNaN(newCount) && newCount > 0 && newCount !== agentCount) {
        agentCount = newCount;
        fitnessHistory = []; // Resetta la storia del fitness
        world = new World(agentCount);
        console.log(`Restarted with ${agentCount} agents.`);
    } else if (isNaN(newCount) || newCount <= 0) {
        console.warn("Invalid agent count specified.");
        newCountInput.value = agentCount.toString(); // Ripristina il valore valido
    }
});