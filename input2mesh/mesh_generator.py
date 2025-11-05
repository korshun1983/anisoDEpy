# mesh_generator.py
import json5
import gmsh
import math
import os
import sys


class ModelParameters:
    """Class to store model parameters from JSON5 configuration"""

    def __init__(self, config_file):
        self.load_config(config_file)
        self.process_parameters()

    def load_config(self, config_file):
        """Load parameters from JSON5 file"""
        with open(config_file, 'r') as f:
            config = json5.load(f)

        self.model_params = config.get('Model', {})
        self.advanced_params = config.get('Advanced', {})
        self.mesh_params = config.get('Mesh', {})

    def process_parameters(self):
        """Process and validate parameters"""
        # Geometry parameters
        self.domain_rx = self.model_params.get('DomainRx', [0.1, 2.0])
        self.domain_ry = self.model_params.get('DomainRy', [0.1, 2.0])
        self.domain_theta = self.model_params.get('DomainTheta', [0, 0])
        self.domain_ecc = self.model_params.get('DomainEcc', [0, 0])
        self.domain_ecc_angle = self.model_params.get('DomainEccAngle', [0, 0])

        # Domain types and properties
        self.domain_types = self.model_params.get('DomainType', ['fluid', 'HTTI'])
        self.num_domains = len(self.domain_types)

        # Additional domain parameters
        self.add_domain_loc = self.model_params.get('AddDomainLoc', 'ext')
        self.add_domain_type = self.model_params.get('AddDomainType', 'abc')
        self.add_domain_L = self.model_params.get('AddDomainL', 1.0)

        # Check if additional domain should be added
        self.has_additional_domain = (self.add_domain_loc == 'ext' and
                                      self.add_domain_type.lower() not in ['none', 'same'])

        # Discretization parameters
        self.domain_nth = self.model_params.get('DomainNth', [12, 12])

        # Mesh parameters
        self.hmax = self.mesh_params.get('hmax', 0.05)
        self.dhmax = self.mesh_params.get('dhmax', 0.25)
        self.ext_boundary_shape = self.mesh_params.get('ext_boundary_shape', 'cir')

        # Calculate total number of computational domains
        self.total_domains = self.num_domains
        if self.has_additional_domain:
            self.total_domains += 1


class MeshGenerator:
    """Class to generate multi-domain mesh using Gmsh"""

    def __init__(self, model_params):
        self.params = model_params
        self.gmsh_initialized = False

        # Mesh data storage
        self.nodes = []
        self.elements = []
        self.boundary_edges = []
        self.domain_tags = []

    def initialize_gmsh(self):
        """Initialize Gmsh environment"""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        self.gmsh_initialized = True

    def finalize_gmsh(self):
        """Finalize Gmsh environment"""
        if self.gmsh_initialized:
            gmsh.finalize()

    def create_elliptical_domain(self, domain_idx, is_outer=False):
        """Create elliptical domain geometry in Gmsh"""
        rx = self.params.domain_rx[domain_idx]
        ry = self.params.domain_ry[domain_idx]
        theta = self.params.domain_theta[domain_idx]
        ecc = self.params.domain_ecc[domain_idx]
        ecc_angle = self.params.domain_ecc_angle[domain_idx]
        num_points = self.params.domain_nth[domain_idx]

        # Calculate center coordinates
        xc = ecc * math.cos(ecc_angle)
        yc = ecc * math.sin(ecc_angle)

        points = []
        curves = []

        # Create points on ellipse
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Transform coordinates based on ellipse parameters and rotation
            x = xc + rx * math.cos(angle) * math.cos(theta) - ry * math.sin(angle) * math.sin(theta)
            y = yc + rx * math.cos(angle) * math.sin(theta) + ry * math.sin(angle) * math.cos(theta)

            point_tag = gmsh.model.occ.addPoint(x, y, 0)
            points.append(point_tag)

        # Create spline curves connecting points
        for i in range(num_points):
            next_point = (i + 1) % num_points
            curve_tag = gmsh.model.occ.addBSpline([points[i], points[next_point]])
            curves.append(curve_tag)

        # Create curve loop
        curve_loop = gmsh.model.occ.addCurveLoop(curves)

        # Create surface
        if is_outer and self.params.ext_boundary_shape == 'rec':
            # Create rectangular boundary for outer domain if specified
            surface_tag = self.create_rectangular_domain(rx, ry, xc, yc, theta)
        else:
            surface_tag = gmsh.model.occ.addPlaneSurface([curve_loop])

        return surface_tag, curves, points

    def create_rectangular_domain(self, rx, ry, xc, yc, theta):
        """Create rectangular domain for outer boundary"""
        # Define rectangle corners
        corners = [
            (-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)
        ]

        points = []
        for corner in corners:
            # Rotate and translate corners
            x_rot = corner[0] * math.cos(theta) - corner[1] * math.sin(theta) + xc
            y_rot = corner[0] * math.sin(theta) + corner[1] * math.cos(theta) + yc
            point_tag = gmsh.model.occ.addPoint(x_rot, y_rot, 0)
            points.append(point_tag)

        # Create lines between points
        lines = []
        for i in range(4):
            next_point = (i + 1) % 4
            line_tag = gmsh.model.occ.addLine(points[i], points[next_point])
            lines.append(line_tag)

        # Create curve loop and surface
        curve_loop = gmsh.model.occ.addCurveLoop(lines)
        surface_tag = gmsh.model.occ.addPlaneSurface([curve_loop])

        return surface_tag

    def create_additional_domain(self):
        """Create additional ABC/PML domain"""
        # Use parameters from last domain with extended size
        last_idx = self.params.num_domains - 1
        rx = self.params.domain_rx[last_idx] + self.params.add_domain_L
        ry = self.params.domain_ry[last_idx] + self.params.add_domain_L
        theta = self.params.domain_theta[last_idx]
        ecc = self.params.domain_ecc[last_idx]
        ecc_angle = self.params.domain_ecc_angle[last_idx]
        num_points = self.params.domain_nth[last_idx]

        xc = ecc * math.cos(ecc_angle)
        yc = ecc * math.sin(ecc_angle)

        # Create outer boundary
        if self.params.ext_boundary_shape == 'rec':
            outer_surface = self.create_rectangular_domain(rx, ry, xc, yc, theta)
        else:
            # Create elliptical outer boundary
            points = []
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                x = xc + rx * math.cos(angle) * math.cos(theta) - ry * math.sin(angle) * math.sin(theta)
                y = yc + rx * math.cos(angle) * math.sin(theta) + ry * math.sin(angle) * math.cos(theta)
                point_tag = gmsh.model.occ.addPoint(x, y, 0)
                points.append(point_tag)

            curves = []
            for i in range(num_points):
                next_point = (i + 1) % num_points
                curve_tag = gmsh.model.occ.addBSpline([points[i], points[next_point]])
                curves.append(curve_tag)

            curve_loop = gmsh.model.occ.addCurveLoop(curves)
            outer_surface = gmsh.model.occ.addPlaneSurface([curve_loop])

        return outer_surface

    def generate_mesh(self):
        """Generate the complete multi-domain mesh"""
        if not self.gmsh_initialized:
            self.initialize_gmsh()

        gmsh.model.add("Waveguide_Mesh")

        domain_surfaces = []
        all_curves = []

        # Create main domains
        for i in range(self.params.num_domains):
            is_outer = (i == self.params.num_domains - 1 and not self.params.has_additional_domain)
            surface_tag, curves, points = self.create_elliptical_domain(i, is_outer)
            domain_surfaces.append(surface_tag)
            all_curves.extend(curves)

        # Create additional domain if needed
        if self.params.has_additional_domain:
            additional_surface = self.create_additional_domain()
            domain_surfaces.append(additional_surface)

        # Synchronize geometry
        gmsh.model.occ.synchronize()

        # Set mesh parameters
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.params.hmax)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", self.params.dhmax)

        # Generate 2D mesh with third-order elements
        gmsh.model.mesh.generate(2)

        # Set element order to 3 (cubic elements)
        gmsh.model.mesh.setOrder(3)

        # Extract mesh data
        self.extract_mesh_data()

        return True

    def extract_mesh_data(self):
        """Extract mesh nodes and elements from Gmsh"""
        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        self.nodes = list(zip(node_coords[0::3], node_coords[1::3]))

        # Get elements (third order triangles - 10 nodes per element)
        element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(2)

        self.elements = []
        for i, elem_type in enumerate(element_types):
            if elem_type == 9:  # 10-node third order triangle
                nodes_per_elem = 10
                num_elems = len(element_tags[i])

                for j in range(num_elems):
                    start_idx = j * nodes_per_elem
                    end_idx = start_idx + nodes_per_elem
                    elem_node_tags = element_nodes[i][start_idx:end_idx]
                    self.elements.append(elem_node_tags)

        # Get boundary edges
        self.extract_boundary_edges()

    def extract_boundary_edges(self):
        """Extract boundary edges information"""
        # Get all edges on boundaries
        edge_entities = gmsh.model.getBoundary([(2, tag) for tag in gmsh.model.getEntities(2)])

        self.boundary_edges = []
        for dim_tag in edge_entities:
            dim, tag = dim_tag
            if dim == 1:  # Edge dimension
                edge_nodes = gmsh.model.mesh.getElementNodes(1, tag)
                # Third order edge has 4 nodes
                if len(edge_nodes) >= 2:
                    self.boundary_edges.append({
                        'nodes': edge_nodes[:4],  # First 4 nodes for cubic element
                        'tag': tag
                    })

    def save_mesh(self, filename):
        """Save mesh to file"""
        if self.gmsh_initialized:
            gmsh.write(filename)

    def visualize_mesh(self):
        """Visualize mesh using Gmsh GUI"""
        if self.gmsh_initialized:
            gmsh.fltk.run()

    def print_mesh_statistics(self):
        """Print mesh statistics"""
        print(f"Mesh Statistics:")
        print(f"Number of nodes: {len(self.nodes)}")
        print(f"Number of elements: {len(self.elements)}")
        print(f"Number of boundary edges: {len(self.boundary_edges)}")
        print(f"Element type: 10-node cubic triangles")

        if self.elements:
            print(f"Nodes per element: {len(self.elements[0])}")

    def generate_mesh_report(self):
        """Generate comprehensive mesh report"""
        report = {
            'total_nodes': len(self.nodes),
            'total_elements': len(self.elements),
            'total_boundary_edges': len(self.boundary_edges),
            'element_order': 3,
            'domain_configuration': {
                'main_domains': self.params.num_domains,
                'additional_domains': 1 if self.params.has_additional_domain else 0,
                'domain_types': self.params.domain_types,
                'has_abc_pml': self.params.has_additional_domain
            },
            'mesh_parameters': {
                'hmax': self.params.hmax,
                'dhmax': self.params.dhmax,
                'boundary_shape': self.params.ext_boundary_shape
            }
        }
        return report


def main():
    """Main function to generate mesh from configuration"""

    # Load model parameters
    config_file = "model_config.json5"
    if not os.path.exists(config_file):
        print(f"Error: Configuration file {config_file} not found!")
        return

    try:
        # Initialize model parameters
        model_params = ModelParameters(config_file)

        # Initialize mesh generator
        mesh_gen = MeshGenerator(model_params)

        print("Generating multi-domain mesh...")
        print(f"Domains: {model_params.total_domains} ({model_params.num_domains} main + "
              f"{1 if model_params.has_additional_domain else 0} additional)")

        # Generate mesh
        success = mesh_gen.generate_mesh()
        if not success:
            print("Error generating mesh!")
            return

        # Print statistics
        mesh_gen.print_mesh_statistics()

        # Save mesh
        output_file = "waveguide_mesh.msh"
        mesh_gen.save_mesh(output_file)
        print(f"Mesh saved to: {output_file}")

        # Generate report
        report = mesh_gen.generate_mesh_report()
        print("\nMesh Generation Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")

        # Visualize if requested
        if model_params.mesh_params.get('output', 'no') == 'yes':
            print("Opening mesh visualization...")
            mesh_gen.visualize_mesh()

    except Exception as e:
        print(f"Error during mesh generation: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        mesh_gen.finalize_gmsh()


if __name__ == "__main__":
    main()