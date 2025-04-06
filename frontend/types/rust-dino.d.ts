declare module 'rust-dino' {
    export function init(): Promise<void>;
    export class World {
        constructor();
        update(dt: number): void;
        jump(): void;
        get_dino_x(): number;
        get_dino_y(): number;
        get_score(): number;
    }
}
