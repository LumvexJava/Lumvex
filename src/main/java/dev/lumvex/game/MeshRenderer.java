package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.PrimitiveComponent;

/** Draws a primitive {@link Mesh} with a {@link Material}. */
public final class MeshRenderer extends Component {
    private final PrimitiveComponent data;

    public MeshRenderer(Mesh mesh, Material material) {
        this(new PrimitiveComponent());
        setMesh(mesh);
        setMaterial(material);
    }

    MeshRenderer(PrimitiveComponent data) { this.data = data; }

    public Mesh mesh() { return Mesh.of(data.shape, data.size); }

    public void setMesh(Mesh mesh) {
        data.shape = mesh.shapeName();
        data.size = mesh.size();
        changed();
    }

    public Material material() { return Material.of(data.materialDef, data.material); }

    public void setMaterial(Material material) {
        data.material = material.assetId();
        data.materialDef = material.toDef();
        changed();
    }

    @Override PrimitiveComponent data() { return data; }
}
