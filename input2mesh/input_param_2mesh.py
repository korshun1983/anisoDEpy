# simple_mesh_test_ide.py
import json
import gmsh
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.tri as tri

# ==================== SETTINGS ====================
CONFIG_FILE = "BakkenB-00.json"

SAVE_MESH = True
SHOW_VISUALIZATION = True
SAVE_PLOT = True
OUTPUT_FILE = None


class SimpleMeshGenerator:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()
        self.frequency_khz = None
        self.process_parameters()
        self.gmsh_initialized = False
        self.mesh_data = {}

    def load_config(self):
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def get_frequency_from_user(self):
        """Get frequency from user input in kHz"""
        print("\n" + "=" * 50)
        print("MESH GENERATION PARAMETERS")
        print("=" * 50)

        model = self.config.get('Model', {})
        f_array = model.get('f_array_range', {})
        f_start = f_array.get('start', 0.5)
        f_end = f_array.get('end', 15.0)

        print(f"Frequency range in config: {f_start} - {f_end} kHz")

        while True:
            try:
                freq_input = input("Enter frequency for mesh generation (kHz): ").strip()
                if not freq_input:
                    print("Using default frequency: 10.0 kHz")
                    return 10.0

                frequency = float(freq_input)
                if frequency <= 0:
                    print("Frequency must be positive. Please try again.")
                    continue

                print(f"Using frequency: {frequency} kHz")
                return frequency

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                exit(0)

    def calculate_wave_velocity(self, domain_type, params):
        """Calculate wave velocity for different domain types"""
        if domain_type == 'fluid':
            density, lambda_param = params[0], params[1]
            # Density in g/cm³, lambda in GPa
            # Convert density to kg/m³: multiply by 1000
            # Convert lambda to Pa: multiply by 1e9
            # velocity = sqrt(lambda / density) in m/s
            density_kg_m3 = density * 1000  # g/cm³ to kg/m³
            lambda_pa = lambda_param * 1e9  # GPa to Pa
            return math.sqrt(lambda_pa / density_kg_m3)
        elif domain_type == 'HTTI':
            density, c11, c13, c33, c44, c66, dip, azimuth = params
            # Use shear wave velocity (Vs) for mesh sizing (usually smaller than Vp)
            # c44 in GPa, density in g/cm³
            density_kg_m3 = density * 1000  # g/cm³ to kg/m³
            c44_pa = c44 * 1e9  # GPa to Pa
            return math.sqrt(c44_pa / density_kg_m3)
        else:
            return 1500.0  # m/s

    def calculate_wavelength_based_hmax(self, frequency_khz):
        """Calculate hmax based on wavelength at specified frequency"""
        model = self.config.get('Model', {})
        domain_params = model.get('DomainParam', [])
        domain_types = model.get('DomainType', [])

        frequency_hz = frequency_khz * 1000.0

        min_velocity = float('inf')
        velocity_info = []

        for i, domain_type in enumerate(domain_types):
            if i < len(domain_params):
                velocity = self.calculate_wave_velocity(domain_type, domain_params[i])
                velocity_info.append(f"{domain_type}: {velocity:.2f} m/s")
                if velocity < min_velocity:
                    min_velocity = velocity

        wavelength = min_velocity / frequency_hz
        hmax = wavelength / 4.0

        print(f"\nWave-based mesh parameters:")
        print(f"  Domain velocities: {', '.join(velocity_info)}")
        print(f"  Min velocity: {min_velocity:.2f} m/s")
        print(f"  Frequency: {frequency_khz} kHz ({frequency_hz:.0f} Hz)")
        print(f"  Wavelength: {wavelength:.6f} m")
        print(f"  hmax: {hmax:.6f} m (wavelength/4)")

        return hmax

    def process_parameters(self):
        model = self.config.get('Model', {})

        self.frequency_khz = self.get_frequency_from_user()

        self.domain_rx = model.get('DomainRx', [0.1, 2.0])
        self.domain_ry = model.get('DomainRy', [0.1, 2.0])
        self.domain_theta = model.get('DomainTheta', [0, 0])
        self.domain_ecc = model.get('DomainEcc', [0, 0])
        self.domain_ecc_angle = model.get('DomainEccAngle', [0, 0])
        self.domain_types = model.get('DomainType', ['fluid', 'HTTI'])
        self.num_domains = len(self.domain_types)

        self.add_domain_loc = model.get('AddDomainLoc', 'ext')
        self.add_domain_type = model.get('AddDomainType', 'abc')
        self.add_domain_L = model.get('AddDomainL', 1.0)
        self.has_additional_domain = (self.add_domain_loc == 'ext' and
                                      self.add_domain_type.lower() not in ['none', 'same'])

        mesh_params = self.config.get('Mesh', {})

        self.hmax = self.calculate_wavelength_based_hmax(self.frequency_khz)
        self.dhmax = 0.3

        self.ext_boundary_shape = mesh_params.get('ext_boundary_shape', 'cir')
        self.domain_nth = model.get('DomainNth', [12, 12])

        print(f"\nModel parameters for {self.frequency_khz} kHz:")
        print(f"  • Domains: {self.num_domains} ({', '.join(self.domain_types)})")
        print(f"  • Radii Rx: {self.domain_rx}")
        print(f"  • Radii Ry: {self.domain_ry}")
        print(f"  • Additional domain: {'Yes' if self.has_additional_domain else 'No'} ({self.add_domain_type})")
        print(f"  • Mesh size: hmax={self.hmax:.6f}, dhmax={self.dhmax}")

    def initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.hmax)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", self.dhmax)
        self.gmsh_initialized = True

    def finalize_gmsh(self):
        if self.gmsh_initialized:
            gmsh.finalize()

    def create_elliptical_domain(self, rx, ry, theta, ecc, ecc_angle, num_points):
        """Create elliptical domain with proper surface"""
        xc = ecc * math.cos(ecc_angle)
        yc = ecc * math.sin(ecc_angle)

        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = xc + rx * math.cos(angle) * math.cos(theta) - ry * math.sin(angle) * math.sin(theta)
            y = yc + rx * math.cos(angle) * math.sin(theta) + ry * math.sin(angle) * math.cos(theta)

            point_tag = gmsh.model.occ.addPoint(x, y, 0)
            points.append(point_tag)

        curves = []
        for i in range(num_points):
            next_idx = (i + 1) % num_points
            curve_tag = gmsh.model.occ.addLine(points[i], points[next_idx])
            curves.append(curve_tag)

        curve_loop = gmsh.model.occ.addCurveLoop(curves)
        surface_tag = gmsh.model.occ.addPlaneSurface([curve_loop])

        return surface_tag, curves

    def create_rectangular_domain(self, rx, ry, theta, ecc, ecc_angle):
        """Create rectangular domain with proper surface"""
        xc = ecc * math.cos(ecc_angle)
        yc = ecc * math.sin(ecc_angle)

        corners = [(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)]

        points = []
        for corner in corners:
            x_rot = corner[0] * math.cos(theta) - corner[1] * math.sin(theta) + xc
            y_rot = corner[0] * math.sin(theta) + corner[1] * math.cos(theta) + yc
            point_tag = gmsh.model.occ.addPoint(x_rot, y_rot, 0)
            points.append(point_tag)

        curves = []
        for i in range(4):
            next_idx = (i + 1) % 4
            curve_tag = gmsh.model.occ.addLine(points[i], points[next_idx])
            curves.append(curve_tag)

        curve_loop = gmsh.model.occ.addCurveLoop(curves)
        surface_tag = gmsh.model.occ.addPlaneSurface([curve_loop])

        return surface_tag, curves

    def generate_mesh(self):
        if not self.gmsh_initialized:
            self.initialize_gmsh()

        gmsh.model.add("Waveguide_Mesh")

        print("\nCreating geometry...")

        # We'll use a simpler approach: create domains sequentially
        # and let Gmsh handle the boolean operations internally

        # Clear any existing geometry
        gmsh.model.occ.removeAllDuplicates()

        # Create main domains as separate surfaces
        domain_surfaces = []
        all_curves = []

        for i in range(self.num_domains):
            rx = self.domain_rx[i]
            ry = self.domain_ry[i]
            theta = self.domain_theta[i]
            ecc = self.domain_ecc[i]
            ecc_angle = self.domain_ecc_angle[i]
            num_points = self.domain_nth[i]

            if i == 0:  # First domain - create as is
                if self.ext_boundary_shape == 'rec' and i == self.num_domains - 1 and not self.has_additional_domain:
                    surface_tag, curves = self.create_rectangular_domain(rx, ry, theta, ecc, ecc_angle)
                else:
                    surface_tag, curves = self.create_elliptical_domain(rx, ry, theta, ecc, ecc_angle, num_points)

                domain_surfaces.append(surface_tag)
                all_curves.append(curves)
                print(f"  Created domain {i + 1}: {self.domain_types[i]}, Rx={rx}, Ry={ry}")

            else:  # Subsequent domains - create as rings
                # For simplicity, we'll create the outer boundary and let Gmsh mesh the annular region
                if self.ext_boundary_shape == 'rec' and i == self.num_domains - 1 and not self.has_additional_domain:
                    surface_tag, curves = self.create_rectangular_domain(rx, ry, theta, ecc, ecc_angle)
                else:
                    surface_tag, curves = self.create_elliptical_domain(rx, ry, theta, ecc, ecc_angle, num_points)

                domain_surfaces.append(surface_tag)
                all_curves.append(curves)
                print(f"  Created domain {i + 1}: {self.domain_types[i]}, Rx={rx}, Ry={ry}")

        # Create additional domain if needed
        if self.has_additional_domain:
            last_idx = self.num_domains - 1
            rx = self.domain_rx[last_idx] + self.add_domain_L
            ry = self.domain_ry[last_idx] + self.add_domain_L
            theta = self.domain_theta[last_idx]
            ecc = self.domain_ecc[last_idx]
            ecc_angle = self.domain_ecc_angle[last_idx]

            if self.ext_boundary_shape == 'rec':
                surface_tag, curves = self.create_rectangular_domain(rx, ry, theta, ecc, ecc_angle)
            else:
                surface_tag, curves = self.create_elliptical_domain(
                    rx, ry, theta, ecc, ecc_angle, self.domain_nth[last_idx]
                )

            domain_surfaces.append(surface_tag)
            all_curves.append(curves)
            print(f"  Created additional domain: {self.add_domain_type}, +{self.add_domain_L}m")

        # Use fragment to combine all geometries and create proper interfaces
        if len(domain_surfaces) > 1:
            print("  Performing fragment operation to combine domains...")
            all_surfaces = [(2, tag) for tag in domain_surfaces]
            gmsh.model.occ.fragment(all_surfaces, [])

        # Synchronize geometry
        gmsh.model.occ.synchronize()

        # Assign physical groups
        for i, surface_tag in enumerate(domain_surfaces):
            if i < self.num_domains:
                gmsh.model.addPhysicalGroup(2, [surface_tag], i + 1)
                gmsh.model.setPhysicalName(2, i + 1, f"Domain_{i + 1}_{self.domain_types[i]}")
                print(f"  Assigned physical group to domain {i + 1}")

        if self.has_additional_domain and len(domain_surfaces) > self.num_domains:
            gmsh.model.addPhysicalGroup(2, [domain_surfaces[self.num_domains]], self.num_domains + 1)
            gmsh.model.setPhysicalName(2, self.num_domains + 1, f"Domain_Additional_{self.add_domain_type}")
            print(f"  Assigned physical group to additional domain")

        # Assign boundary physical groups
        if all_curves:
            # Get all curves after fragmentation
            all_boundary_curves = []
            for curves in all_curves:
                all_boundary_curves.extend(curves)

            # Remove duplicates
            unique_curves = list(set(all_boundary_curves))
            gmsh.model.addPhysicalGroup(1, unique_curves, 9999)
            gmsh.model.setPhysicalName(1, 9999, "All_Boundaries")
            print(f"  Assigned physical group to {len(unique_curves)} boundary curves")

        # Set mesh parameters
        gmsh.option.setNumber("Mesh.ElementOrder", 3)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)

        print("Generating mesh...")
        gmsh.model.mesh.generate(2)

        # Check what element types were generated
        element_types, _, _ = gmsh.model.mesh.getElements(2)
        print(f"Generated element types: {element_types}")

        self.extract_mesh_data()

        return True

    def extract_mesh_data(self):
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

        # Create mapping from node tag to index
        node_tag_to_index = {}
        for idx, tag in enumerate(nodeTags):
            node_tag_to_index[tag] = idx

        self.mesh_data['nodes'] = np.array(nodeCoords).reshape(-1, 3)[:, :2]
        self.mesh_data['node_tags'] = nodeTags
        self.mesh_data['node_tag_to_index'] = node_tag_to_index

        elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

        total_nodes = len(self.mesh_data['nodes'])
        total_elements = 0

        self.mesh_data['elements'] = {}
        self.mesh_data['element_domains'] = {}

        print(f"Available element types: {elementTypes}")

        for i in range(len(elementTypes)):
            elem_type = elementTypes[i]
            elem_tags = elementTags[i]
            elem_nodes = nodeTags[i]

            total_elements += len(elem_tags)

            # Convert node tags to indices using mapping
            node_indices = [node_tag_to_index[tag] for tag in elem_nodes]

            nodes_per_elem = self.get_nodes_per_element(elem_type)

            self.mesh_data['elements'][elem_type] = {
                'tags': elem_tags,
                'nodes': np.array(node_indices).reshape(-1, nodes_per_elem)
            }

            self.mesh_data['element_domains'][elem_type] = np.zeros(len(elem_tags), dtype=int)

            print(f"  Element type {elem_type}: {len(elem_tags)} elements, {nodes_per_elem} nodes per element")

        self.assign_element_domains()

        print(f"  Nodes: {total_nodes}")
        print(f"  Elements: {total_elements}")

        self.extract_boundary_elements()

    def get_nodes_per_element(self, elem_type):
        """Return number of nodes for different element types"""
        if elem_type == 2:  # 3-node triangle (linear)
            return 3
        elif elem_type == 9:  # 6-node second order triangle
            return 6
        elif elem_type == 21:  # 10-node third order triangle
            return 10
        elif elem_type == 3:  # 4-node quadrangle
            return 4
        else:
            print(f"  Warning: Unknown element type {elem_type}, assuming 3 nodes")
            return 3

    def assign_element_domains(self):
        """Assign domains to elements via physical groups - FIXED VERSION"""
        # Get all physical groups
        physical_groups = gmsh.model.getPhysicalGroups()

        print(f"Found {len(physical_groups)} physical groups")

        for dim, tag in physical_groups:
            if dim == 2:  # Surface elements (domains)
                print(f"Processing physical group {tag} (dim={dim})")

                # Get entity tags for this physical group
                entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                print(f"  Entity tags: {entity_tags}")

                # entity_tags is a list of entity tags (not tuples)
                for entity_tag in entity_tags:
                    print(f"  Entity: dim={dim}, tag={entity_tag}")

                    # Get elements for this entity
                    try:
                        elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim, entity_tag)

                        for i in range(len(elemTypes)):
                            elem_type = elemTypes[i]
                            if elem_type in self.mesh_data['element_domains']:
                                # Find element indices in the general array
                                for elem_tag in elemTags[i]:
                                    idx = np.where(self.mesh_data['elements'][elem_type]['tags'] == elem_tag)[0]
                                    if len(idx) > 0:
                                        self.mesh_data['element_domains'][elem_type][idx[0]] = tag
                                        print(f"    Assigned domain {tag} to element {elem_tag} of type {elem_type}")
                    except Exception as e:
                        print(f"    Error getting elements for entity {entity_tag}: {e}")

    def extract_boundary_elements(self):
        boundary_elementTypes, boundary_elementTags, boundary_nodeTags = gmsh.model.mesh.getElements(1)

        self.mesh_data['boundary_elements'] = {}

        for i in range(len(boundary_elementTypes)):
            elem_type = boundary_elementTypes[i]
            elem_tags = boundary_elementTags[i]
            elem_nodes = boundary_nodeTags[i]

            # Convert node tags to indices using mapping
            node_indices = [self.mesh_data['node_tag_to_index'][tag] for tag in elem_nodes]

            self.mesh_data['boundary_elements'][elem_type] = {
                'tags': elem_tags,
                'nodes': np.array(node_indices).reshape(-1, self.get_nodes_per_boundary_element(elem_type))
            }

        print(f"  Boundary elements: {sum(len(data['tags']) for data in self.mesh_data['boundary_elements'].values())}")

    def get_nodes_per_boundary_element(self, elem_type):
        if elem_type == 1:
            return 2
        elif elem_type == 8:
            return 3
        elif elem_type == 26:
            return 4
        else:
            return 2

    def save_mesh(self, output_file=None):
        if output_file is None:
            base_name = os.path.splitext(self.config_file)[0]
            freq_str = f"{self.frequency_khz:.1f}".replace('.', 'p')
            output_file = f"{base_name}_mesh_{freq_str}kHz.msh"

        gmsh.write(output_file)
        print(f"Mesh saved to: {output_file}")
        return output_file

    def load_and_display_gmsh_mesh(self, mesh_file):
        if not self.gmsh_initialized:
            self.initialize_gmsh()

        gmsh.open(mesh_file)
        print("Displaying mesh in Gmsh GUI...")
        gmsh.fltk.run()

    def extract_triangulation_data(self):
        """Extract triangulation data for triplot visualization"""
        # Try different element types in order of preference
        element_types_to_try = [21, 9, 2]  # 10-node, 6-node, 3-node triangles

        for elem_type in element_types_to_try:
            if elem_type in self.mesh_data['elements']:
                elements = self.mesh_data['elements'][elem_type]['nodes']
                all_nodes = self.mesh_data['nodes']

                print(f"Using element type {elem_type} for triangulation")

                # For all triangle types, the first 3 nodes are the corners
                linear_triangles = elements[:, :3]  # Already 0-based indices

                # Identify additional nodes (all nodes not used as corners in linear triangles)
                corner_node_indices = set(linear_triangles.flatten())
                all_node_indices = set(range(len(all_nodes)))
                additional_node_indices = list(all_node_indices - corner_node_indices)

                # Create triangulation object
                triangulation = tri.Triangulation(all_nodes[:, 0], all_nodes[:, 1], linear_triangles)

                return triangulation, linear_triangles, additional_node_indices, elem_type

        print("No suitable triangle elements found for triangulation")
        return None, None, None, None

    def visualize_with_triplot(self):
        """Simplified visualization using triplot - only one figure with all nodes"""
        if not self.mesh_data or 'elements' not in self.mesh_data:
            print("No mesh data for visualization")
            return

        print("Creating triplot visualization...")

        # Extract triangulation data
        triangulation, linear_triangles, additional_node_indices, elem_type = self.extract_triangulation_data()

        if triangulation is None:
            print("Failed to extract triangulation data")
            return

        all_nodes = self.mesh_data['nodes']

        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Create triplot
        ax.triplot(triangulation, 'b-', lw=0.8, alpha=0.8)

        # Plot all nodes
        ax.plot(all_nodes[:, 0], all_nodes[:, 1], 'go', markersize=3, alpha=0.8, label='All nodes')

        # Highlight additional nodes
        if additional_node_indices:
            additional_nodes = all_nodes[additional_node_indices]
            ax.plot(additional_nodes[:, 0], additional_nodes[:, 1], 'ro', markersize=4, alpha=0.8,
                    label='Additional nodes')

        # Add boundary elements
        if 'boundary_elements' in self.mesh_data:
            for elem_type_boundary, elements in self.mesh_data['boundary_elements'].items():
                for elem_nodes in elements['nodes']:
                    coords = all_nodes[elem_nodes]
                    x_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8)

        ax.set_title(f'Waveguide Mesh - {self.frequency_khz} kHz\n(Triplot with Cubic Elements)', fontsize=14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()

        # Add mesh information
        total_nodes = len(all_nodes)
        total_elements = sum(len(elements['tags']) for elements in self.mesh_data['elements'].values())
        linear_tri_count = len(linear_triangles) if linear_triangles is not None else 0

        info_text = f'Nodes: {total_nodes}\nElements: {total_elements}\nLinear triangles: {linear_tri_count}\nhmax: {self.hmax:.6f} m'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

        plt.tight_layout()

        if SAVE_PLOT:
            freq_str = f"{self.frequency_khz:.1f}".replace('.', 'p')
            plot_file = os.path.splitext(self.config_file)[0] + f'_triplot_{freq_str}kHz.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Triplot visualization saved to: {plot_file}")

        if SHOW_VISUALIZATION:
            plt.show()

        return fig, ax


def main():
    print("=" * 50)
    print("WAVEGUIDE MESH GENERATOR WITH CORRECTED UNITS")
    print("=" * 50)

    mesh_generator = None
    mesh_file_path = None

    try:
        if not os.path.exists(CONFIG_FILE):
            print(f"ERROR: File '{CONFIG_FILE}' not found!")
            return

        mesh_generator = SimpleMeshGenerator(CONFIG_FILE)

        success = mesh_generator.generate_mesh()
        if not success:
            print("Error generating mesh!")
            return

        if SAVE_MESH:
            mesh_file_path = mesh_generator.save_mesh(OUTPUT_FILE)

        # Visualize with triplot
        if SHOW_VISUALIZATION:
            print("\nGenerating visualization...")
            mesh_generator.visualize_with_triplot()

            # Gmsh visualization (optional)
            show_gmsh = input("\nShow mesh in Gmsh GUI? (y/n): ").strip().lower()
            if show_gmsh in ['y', 'yes'] and mesh_file_path and os.path.exists(mesh_file_path):
                print("Opening Gmsh visualization...")
                mesh_generator.load_and_display_gmsh_mesh(mesh_file_path)

        print("\n" + "=" * 50)
        print("GENERATION COMPLETED SUCCESSFULLY!")
        if SAVE_MESH:
            print(f"Mesh file: {mesh_file_path}")
        print(f"Frequency: {mesh_generator.frequency_khz} kHz")
        print(f"Mesh size (hmax): {mesh_generator.hmax:.6f} m")
        print("=" * 50)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        if mesh_generator:
            mesh_generator.finalize_gmsh()


if __name__ == "__main__":
    main()