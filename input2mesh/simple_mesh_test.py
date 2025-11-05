# simple_mesh_test_ide.py
import pyjson5 as json5
import gmsh
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.tri as tri

# ==================== SETTINGS ====================
CONFIG_FILE = "BakkenB-00.json5"

SAVE_MESH = True
SHOW_VISUALIZATION = True
SAVE_PLOT = True
OUTPUT_FILE = None


class SimpleMeshGenerator:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()
        self.frequency_khz = None  # Will be set by user input
        self.process_parameters()
        self.gmsh_initialized = False
        self.mesh_data = {}

    def load_config(self):
        with open(self.config_file, 'r') as f:
            return json5.load(f)

    def get_frequency_from_user(self):
        """Get frequency from user input in kHz"""
        print("\n" + "=" * 50)
        print("MESH GENERATION PARAMETERS")
        print("=" * 50)

        # Show frequency range from config for reference
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
            # Longitudinal wave velocity in fluid
            return math.sqrt(lambda_param / density)*1e3
        elif domain_type == 'HTTI':
            density, c11, c13, c33, c44, c66, dip, azimuth = params
            # Use shear wave velocity (Vs) for mesh sizing (usually smaller than Vp)
            return math.sqrt(c44 / density)*1e3
        else:
            # Default velocity if domain type is unknown
            return 1500.0  # m/s

    def calculate_wavelength_based_hmax(self, frequency_khz):
        """Calculate hmax based on wavelength at specified frequency"""
        model = self.config.get('Model', {})
        domain_params = model.get('DomainParam', [])
        domain_types = model.get('DomainType', [])

        # Convert kHz to Hz
        frequency_hz = frequency_khz * 1000.0

        # Find minimum wave velocity across all domains
        min_velocity = float('inf')
        velocity_info = []

        for i, domain_type in enumerate(domain_types):
            if i < len(domain_params):
                velocity = self.calculate_wave_velocity(domain_type, domain_params[i])
                velocity_info.append(f"{domain_type}: {velocity:.2f} m/s")
                if velocity < min_velocity:
                    min_velocity = velocity

        # Calculate wavelength at specified frequency
        wavelength = min_velocity / frequency_hz

        # hmax = wavelength / 4
        hmax = wavelength / 4.0

        print(f"\nWave-based mesh parameters:")
        print(f"  Domain velocities: {', '.join(velocity_info)}")
        print(f"  Min velocity: {min_velocity:.2f} m/s")
        print(f"  Frequency: {frequency_khz} kHz ({frequency_hz:.0f} Hz)")
        print(f"  Wavelength: {wavelength:.6f} m")
        print(f"  hmax: {hmax:.6f} m (wavelength/4)")

        return hmax

    def process_parameters(self):
        """Process parameters with user-defined frequency"""
        model = self.config.get('Model', {})

        # Get frequency from user
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

        # Calculate hmax based on user-defined frequency
        self.hmax = self.calculate_wavelength_based_hmax(self.frequency_khz)

        # Set dhmax to 0.3 as requested
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

    def create_elliptical_boundary(self, rx, ry, theta, ecc, ecc_angle, num_points, domain_idx):
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

        curve_physical_tag = 1000 + domain_idx
        gmsh.model.addPhysicalGroup(1, curves, curve_physical_tag)
        gmsh.model.setPhysicalName(1, curve_physical_tag, f"Boundary_Domain_{domain_idx}")

        return surface_tag, curves

    def create_rectangular_boundary(self, rx, ry, theta, ecc, ecc_angle, domain_idx):
        xc = ecc * math.cos(ecc_angle)
        yc = ecc * math.sin(ecc_angle)

        corners = [(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)]

        points = []
        for corner in corners:
            x_rot = corner[0] * math.cos(theta) - corner[1] * math.sin(theta) + xc
            y_rot = corner[0] * math.sin(theta) + corner[1] * math.cos(theta) + yc
            point_tag = gmsh.model.occ.addPoint(x_rot, y_rot, 0)
            points.append(point_tag)

        lines = []
        for i in range(4):
            next_idx = (i + 1) % 4
            line_tag = gmsh.model.occ.addLine(points[i], points[next_idx])
            lines.append(line_tag)

        curve_loop = gmsh.model.occ.addCurveLoop(lines)
        surface_tag = gmsh.model.occ.addPlaneSurface([curve_loop])

        curve_physical_tag = 1000 + domain_idx
        gmsh.model.addPhysicalGroup(1, lines, curve_physical_tag)
        gmsh.model.setPhysicalName(1, curve_physical_tag, f"Boundary_Domain_{domain_idx}")

        return surface_tag, lines

    def generate_mesh(self):
        if not self.gmsh_initialized:
            self.initialize_gmsh()

        gmsh.model.add("Waveguide_Mesh")

        print("\nCreating geometry...")

        domain_surfaces = []
        all_boundary_curves = []

        for i in range(self.num_domains):
            rx = self.domain_rx[i]
            ry = self.domain_ry[i]
            theta = self.domain_theta[i]
            ecc = self.domain_ecc[i]
            ecc_angle = self.domain_ecc_angle[i]
            num_points = self.domain_nth[i]

            is_outer = (i == self.num_domains - 1 and not self.has_additional_domain)

            if is_outer and self.ext_boundary_shape == 'rec':
                surface_tag, curves = self.create_rectangular_boundary(rx, ry, theta, ecc, ecc_angle, i + 1)
            else:
                surface_tag, curves = self.create_elliptical_boundary(rx, ry, theta, ecc, ecc_angle, num_points, i + 1)

            domain_surfaces.append(surface_tag)
            all_boundary_curves.extend(curves)

            gmsh.model.addPhysicalGroup(2, [surface_tag], i + 1)
            gmsh.model.setPhysicalName(2, i + 1, f"Domain_{i + 1}_{self.domain_types[i]}")

            print(f"  Created domain {i + 1}: {self.domain_types[i]}, Rx={rx}, Ry={ry}")

        if self.has_additional_domain:
            last_idx = self.num_domains - 1
            rx = self.domain_rx[last_idx] + self.add_domain_L
            ry = self.domain_ry[last_idx] + self.add_domain_L
            theta = self.domain_theta[last_idx]
            ecc = self.domain_ecc[last_idx]
            ecc_angle = self.domain_ecc_angle[last_idx]

            domain_idx = self.num_domains + 1

            if self.ext_boundary_shape == 'rec':
                surface_tag, curves = self.create_rectangular_boundary(rx, ry, theta, ecc, ecc_angle, domain_idx)
            else:
                surface_tag, curves = self.create_elliptical_boundary(
                    rx, ry, theta, ecc, ecc_angle, self.domain_nth[last_idx], domain_idx
                )

            domain_surfaces.append(surface_tag)
            all_boundary_curves.extend(curves)

            gmsh.model.addPhysicalGroup(2, [surface_tag], domain_idx)
            gmsh.model.setPhysicalName(2, domain_idx, f"Domain_Additional_{self.add_domain_type}")
            print(f"  Created additional domain: {self.add_domain_type}, +{self.add_domain_L}m")

        if all_boundary_curves:
            gmsh.model.addPhysicalGroup(1, all_boundary_curves, 9999)
            gmsh.model.setPhysicalName(1, 9999, "All_Boundaries")

        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.ElementOrder", 3)

        print("Generating mesh...")
        gmsh.model.mesh.generate(2)

        self.extract_mesh_data()

        return True

    def extract_mesh_data(self):
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        self.mesh_data['nodes'] = np.array(nodeCoords).reshape(-1, 3)[:, :2]
        self.mesh_data['node_tags'] = nodeTags

        elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

        total_nodes = len(self.mesh_data['nodes'])
        total_elements = 0

        self.mesh_data['elements'] = {}
        self.mesh_data['element_domains'] = {}

        for i in range(len(elementTypes)):
            elem_type = elementTypes[i]
            elem_tags = elementTags[i]
            elem_nodes = nodeTags[i]

            total_elements += len(elem_tags)

            self.mesh_data['elements'][elem_type] = {
                'tags': elem_tags,
                'nodes': np.array(elem_nodes).reshape(-1, self.get_nodes_per_element(elem_type))
            }

            self.mesh_data['element_domains'][elem_type] = np.zeros(len(elem_tags), dtype=int)

            if elem_type == 9:
                print(f"  Element type: 10-node triangles (cubic)")

        self.assign_element_domains()

        print(f"  Nodes: {total_nodes}")
        print(f"  Elements: {total_elements}")

        self.extract_boundary_elements()

    def get_nodes_per_element(self, elem_type):
        if elem_type == 2:
            return 3
        elif elem_type == 9:
            return 10
        else:
            return 3

    def assign_element_domains(self):
        physical_groups = gmsh.model.getPhysicalGroups()

        for dim, tag in physical_groups:
            if dim == 2:
                elemTypes, elemTags, _ = gmsh.model.mesh.getElementsForPhysicalGroup(dim, tag)

                for i in range(len(elemTypes)):
                    elem_type = elemTypes[i]
                    if elem_type in self.mesh_data['element_domains']:
                        for elem_tag in elemTags[i]:
                            idx = np.where(self.mesh_data['elements'][elem_type]['tags'] == elem_tag)[0]
                            if len(idx) > 0:
                                self.mesh_data['element_domains'][elem_type][idx[0]] = tag

    def extract_boundary_elements(self):
        boundary_elementTypes, boundary_elementTags, boundary_nodeTags = gmsh.model.mesh.getElements(1)

        self.mesh_data['boundary_elements'] = {}

        for i in range(len(boundary_elementTypes)):
            elem_type = boundary_elementTypes[i]
            elem_tags = boundary_elementTags[i]
            elem_nodes = boundary_nodeTags[i]

            self.mesh_data['boundary_elements'][elem_type] = {
                'tags': elem_tags,
                'nodes': np.array(elem_nodes).reshape(-1, self.get_nodes_per_boundary_element(elem_type))
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
            # Include frequency in filename
            freq_str = f"{self.frequency_khz:.1f}".replace('.', 'p')
            output_file = f"{base_name}_mesh_{freq_str}kHz.msh"

        gmsh.write(output_file)
        print(f"Mesh saved to: {output_file}")
        return output_file

    def load_and_display_gmsh_mesh(self, mesh_file):
        """Load and display mesh using Gmsh's built-in visualization"""
        if not self.gmsh_initialized:
            self.initialize_gmsh()

        gmsh.open(mesh_file)
        print("Displaying mesh in Gmsh GUI...")
        gmsh.fltk.run()

    def extract_linear_triangles(self):
        """Extract linear triangles from cubic elements for triplot visualization"""
        if 9 not in self.mesh_data['elements']:
            print("No cubic elements found for extraction")
            return None, None

        cubic_elements = self.mesh_data['elements'][9]['nodes']
        all_nodes = self.mesh_data['nodes']

        # For cubic triangles (10 nodes), the first 3 nodes are the corners
        linear_triangles = cubic_elements[:, :3] - 1  # Convert to 0-based indexing

        # Identify additional nodes (all nodes not used as corners in linear triangles)
        corner_node_indices = set(linear_triangles.flatten())
        all_node_indices = set(range(len(all_nodes)))
        additional_node_indices = list(all_node_indices - corner_node_indices)

        return linear_triangles, additional_node_indices

    def visualize_with_triplot(self):
        """Visualize mesh using triplot for linear triangles and separate additional nodes"""
        if not self.mesh_data or 'elements' not in self.mesh_data:
            print("No mesh data for visualization")
            return

        print("Creating triplot visualization...")

        # Extract linear triangles from cubic elements
        linear_triangles, additional_node_indices = self.extract_linear_triangles()

        if linear_triangles is None:
            print("Failed to extract linear triangles")
            return

        all_nodes = self.mesh_data['nodes']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Triplot with additional nodes
        if len(linear_triangles) > 0:
            triangulation = tri.Triangulation(all_nodes[:, 0], all_nodes[:, 1], linear_triangles)
            ax1.triplot(triangulation, 'b-', lw=0.5, alpha=0.8)
            ax1.plot(all_nodes[:, 0], all_nodes[:, 1], 'go', markersize=2, alpha=0.6)

            # Plot additional nodes in red
            if additional_node_indices:
                additional_nodes = all_nodes[list(additional_node_indices)]
                ax1.plot(additional_nodes[:, 0], additional_nodes[:, 1], 'ro', markersize=3, alpha=0.8)

        ax1.set_title(
            f'Triplot Visualization - {self.frequency_khz} kHz\n(Blue: Linear triangles, Red: Additional nodes)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Plot 2: Original visualization for comparison
        for elem_type, elements in self.mesh_data['elements'].items():
            if elem_type == 9:
                for elem_nodes in elements['nodes']:
                    corner_indices = [elem_nodes[0] - 1, elem_nodes[1] - 1, elem_nodes[2] - 1, elem_nodes[0] - 1]
                    corner_coords = all_nodes[corner_indices]
                    ax2.plot(corner_coords[:, 0], corner_coords[:, 1], 'b-', linewidth=0.8, alpha=0.8)

        ax2.plot(all_nodes[:, 0], all_nodes[:, 1], 'go', markersize=2, alpha=0.6)

        if 'boundary_elements' in self.mesh_data:
            for elem_type, elements in self.mesh_data['boundary_elements'].items():
                for elem_nodes in elements['nodes']:
                    node_indices = [node_idx - 1 for node_idx in elem_nodes]
                    coords = all_nodes[node_indices]
                    ax2.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2, alpha=0.8)

        ax2.set_title(f'Original Visualization - {self.frequency_khz} kHz\n(Blue: Elements, Red: Boundaries)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Add information text
        total_nodes = len(all_nodes)
        total_elements = sum(len(elements['tags']) for elements in self.mesh_data['elements'].values())
        linear_tri_count = len(linear_triangles) if linear_triangles is not None else 0
        additional_node_count = len(additional_node_indices) if additional_node_indices else 0

        info_text = f'Total nodes: {total_nodes}\nLinear triangles: {linear_tri_count}\nAdditional nodes: {additional_node_count}\nFrequency: {self.frequency_khz} kHz\nhmax: {self.hmax:.6f} m'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

        plt.tight_layout()

        if SAVE_PLOT:
            freq_str = f"{self.frequency_khz:.1f}".replace('.', 'p')
            plot_file = os.path.splitext(self.config_file)[0] + f'_mesh_{freq_str}kHz.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Triplot visualization saved to: {plot_file}")

        if SHOW_VISUALIZATION:
            plt.show()

        return fig, (ax1, ax2)


def main():
    print("=" * 50)
    print("WAVEGUIDE MESH GENERATOR WITH FREQUENCY INPUT")
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

        # Visualize with both methods
        if SHOW_VISUALIZATION:
            # 1. Triplot visualization
            mesh_generator.visualize_with_triplot()

            # 2. Gmsh visualization (optional)
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