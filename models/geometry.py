class Layer:
    def __init__(self, outer_radius, material, inner_radius=None, n_points=12):
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius) if inner_radius is not None else 0.0
        self.material = material
        self.n_points = int(n_points)

    @property
    def thickness(self):
        return self.outer_radius - self.inner_radius

    def __repr__(self):
        return (f"Layer(outer={self.outer_radius}, inner={self.inner_radius}, "
                f"material={self.material}, n_points={self.n_points})")


class Geometry:
    def __init__(self, layers):
        self.layers = layers
        self._validate()

    def _validate(self):
        prev = 0.0
        for i, layer in enumerate(self.layers):
            if abs(layer.inner_radius - prev) > 1e-12:
                raise ValueError(f"Нестыковка радиусов на слое {i}")
            prev = layer.outer_radius

    def num_layers(self):
        return len(self.layers)

    def radii_list(self):
        return [layer.outer_radius for layer in self.layers]

    def get_layer(self, idx):
        return self.layers[idx]