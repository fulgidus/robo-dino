import init, { World } from './rust/rust_dino.js';

const canvas = document.getElementById('game') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
let world: World;
let lastTime = 0;

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const dinoX = world.get_dino_x();
    const dinoY = world.get_dino_y();
    const score = world.get_score();

    // Suolo
    ctx.fillStyle = 'brown';
    ctx.fillRect(0, canvas.height - 20, canvas.width, 20);

    // Dino
    ctx.fillStyle = 'green';
    const dinoCanvasY = canvas.height - 20 - dinoY - 20;
    ctx.fillRect(dinoX, dinoCanvasY, 20, 20);

    // Ostacoli
    for (let i = 0; i < world.get_obstacle_count(); i++) {
        const x = world.get_obstacle_x(i);
        const obsY = canvas.height - 20 - 30;
        ctx.fillStyle = 'red';
        ctx.fillRect(x, obsY, 20, 30);
    }

    // Score
    ctx.fillStyle = 'black';
    ctx.font = '16px monospace';
    ctx.fillText(`Score: ${score}`, 10, 20);
}


function loop(timestamp: number) {
    const dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    world.update(dt);
    draw();
    requestAnimationFrame(loop);
}

function setup() {
    canvas.addEventListener('click', () => world.jump());
    requestAnimationFrame(loop);
}

init().then(() => {
    world = new World();
    setup();
});
