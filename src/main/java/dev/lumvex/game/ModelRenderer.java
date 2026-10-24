package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.ModelComponent;

/** Draws a glTF model of the project, e.g. {@code "models/chair.glb"}, with its own materials. */
public final class ModelRenderer extends Component {
    private final ModelComponent data;

    public ModelRenderer(String asset) { this(new ModelComponent(asset)); }

    ModelRenderer(ModelComponent data) { this.data = data; }

    public String assetId() { return data.model; }

    /** Restricts the model to the nodes whose name starts with {@code prefix}, or shows all nodes with null. */
    public void setNode(String prefix) {
        data.node = prefix;
        changed();
    }

    /** Whether the model is moved so that its bounding box is centered on the object's origin. */
    public void setRecentered(boolean recentered) {
        data.recenter = recentered;
        changed();
    }

    /** Replaces all materials of the model by a material asset, or restores them with null. */
    public void setMaterialOverride(Material asset) {
        if (asset != null && asset.assetId() == null) throw new IllegalArgumentException("models take material assets only");
        data.material = asset == null ? null : asset.assetId();
        changed();
    }

    /** Radiance of the candle flames in the model; 0 puts them out. */
    public void setFlameIntensity(float intensity) {
        data.flameRadiance = intensity;
        changed();
    }

    public float flameIntensity() { return data.flameRadiance; }

    @Override ModelComponent data() { return data; }
}
