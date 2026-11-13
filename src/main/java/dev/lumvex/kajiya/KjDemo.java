package dev.lumvex.kajiya;

//DEV

import dev.lumvex.game.*;

public final class KjDemo {
    public static void main(String[] args) throws Exception {
        Game game = new Game("Kajiya pipeline demo");
        game.setWindowSize(1280, 720);
        game.addScene(new DemoScene());
        game.start();
    }

    public static final class DemoScene extends Scene {
        @Override
        protected void build() {
            lighting().setSun(new Vec3(0.6f, 0.5f, 0.4f), Color.WHITE, 3);
            add(new FloorObject());
            add(new SpinningCube(new Vec3(0, 0.6f, 0)));
            add(new Viewer(new Vec3(0, 1.4f, 3.6f)));
        }
    }

    public static final class FloorObject extends GameObject {
        public FloorObject() {
            super("Floor");
            addComponent(new MeshRenderer(Mesh.plane(40), Material.color(Color.hex("#999999"))));
        }
    }

    public static final class Viewer extends GameObject {
        public Viewer(Vec3 position) {
            super("Viewer");
            setPosition(position);
            addComponent(new Camera());
        }
    }

    public static final class SpinningCube extends GameObject {
        public SpinningCube(Vec3 position) {
            super("Cube");
            setPosition(position);
            Material gold = Material.metal(Color.hex("#FFC857")).withRoughness(0.2f);
            addComponent(new MeshRenderer(Mesh.cube(0.8f), gold));
            addScript(new Spin(45));
        }
    }

    public static final class Spin extends ObjectScript {
        private final float degreesPerSecond;

        public Spin(float degreesPerSecond) { this.degreesPerSecond = degreesPerSecond; }

        @Override
        public void onUpdate(float dt) { rotate(0, degreesPerSecond * dt, 0); }
    }
}
