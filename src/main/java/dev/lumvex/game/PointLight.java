package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.LightComponent;

/** A small spherical light at the object's position, such as a bulb or a candle. */
public final class PointLight extends Component {
    private final LightComponent data;

    /** A light of the given color; {@code intensity} is the radiance of its surface. */
    public PointLight(Color color, float intensity) {
        this(new LightComponent());
        data.color = color.rgb();
        data.intensity = intensity;
    }

    PointLight(LightComponent data) { this.data = data; }

    public Color color() { return Color.of(data.color); }

    public void setColor(Color color) {
        data.color = color.rgb();
        changed();
    }

    public float intensity() { return data.intensity; }

    /** Sets the radiance; 0 switches the light off. */
    public void setIntensity(float intensity) {
        data.intensity = intensity;
        changed();
    }

    /** Radius of the emitting sphere in meters. Larger lights cast softer shadows. */
    public void setRadius(float radius) {
        data.radius = radius;
        changed();
    }

    @Override LightComponent data() { return data; }
}
