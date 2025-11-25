import gmsh
import numpy as np
from .mesh_container import MeshContainer


def get_min_velocity(model_dict):
    """Calculate minimum wave velocity in the model for mesh sizing."""
    m = model_dict['Model']
    min_velocity = float('inf')

    for i, domain_type in enumerate(m['DomainType']):
        params = m['DomainParam'][i]
        if domain_type.lower() == 'fluid':
            # For fluid: [density, longitudinal_velocity] in km/s
            velocity = params[1] * 1000  # Convert km/s to m/s
        else:
            # For solid materials: calculate shear wave velocity
            # Assuming params: [density, C11, C33, C13, C55, C66, ...]
            # Convert from GPa to Pa and g/cm³ to kg/m³
            density = params[0] * 1000  # g/cm³ to kg/m³
            C66 = params[5] * 1e9  # GPa to Pa
            velocity = np.sqrt(C66 / density)

        if velocity < min_velocity:
            min_velocity = velocity

    return min_velocity


def debug_domain_calculation(model_dict, frequency, min_velocity):
    """Debug function to show domain radius calculations."""
    m = model_dict['Model']

    print(f"\n{'=' * 60}")
    print(f"DEBUG: Domain Calculation for {frequency} kHz")
    print(f"{'=' * 60}")

    # Show original parameters
    print(f"Original parameters:")
    print(f"  DomainRx: {m['DomainRx']}")
    print(f"  DomainRy: {m['DomainRy']}")
    print(f"  AddDomainLoc: {m['AddDomainLoc']}")
    print(f"  AddDomainType: {m['AddDomainType']}")
    print(f"  AddDomainL: {m['AddDomainL']}")
    print(f"  LDomain_in_LSH: {m.get('LDomain_in_LSH', 'none')}")

    # Calculate wavelength and show calculation parameters
    wavelength = min_velocity / (frequency * 1000)  # frequency in kHz to Hz
    print(f"\nCalculation parameters:")
    print(f"  Frequency: {frequency} kHz")
    print(f"  Min velocity: {min_velocity:.0f} m/s")
    print(f"  Wavelength: {wavelength:.6f} m")

    # Calculate what the final radii should be
    domain_rx = m['DomainRx'].copy()
    domain_ry = m['DomainRy'].copy()

    # Apply the same logic as in the main function
    if 'AddDomainType' in m and m['AddDomainType'].lower() != 'none':
        if m['AddDomainLoc'].lower() == 'ext':
            abc_rx = domain_rx[-1] + m['AddDomainL']
            abc_ry = domain_ry[-1] + m['AddDomainL']
            domain_rx = np.append(domain_rx, abc_rx)
            domain_ry = np.append(domain_ry, abc_ry)

    print(f"\nFinal domain radii (current logic):")
    for i, (rx, ry) in enumerate(zip(domain_rx, domain_ry)):
        domain_type = "main" if i < len(m['DomainType']) else "ABC"
        print(f"  Domain {i + 1} ({domain_type}): Rx = {rx:.3f}, Ry = {ry:.3f}")

    # Show what the radii WOULD be with LDomain_in_LSH='yes'
    if m.get('LDomain_in_LSH', 'none').lower() == 'yes':
        domain_rx_scaled = m['DomainRx'].copy()
        domain_ry_scaled = m['DomainRy'].copy()

        # Scale the last main domain
        domain_rx_scaled[-1] = domain_rx_scaled[-2] + domain_rx_scaled[-1] * wavelength
        domain_ry_scaled[-1] = domain_ry_scaled[-2] + domain_ry_scaled[-1] * wavelength

        # Add ABC domain with scaled length
        abc_rx_scaled = domain_rx_scaled[-1] + m['AddDomainL'] * wavelength
        abc_ry_scaled = domain_ry_scaled[-1] + m['AddDomainL'] * wavelength
        domain_rx_scaled = np.append(domain_rx_scaled, abc_rx_scaled)
        domain_ry_scaled = np.append(domain_ry_scaled, abc_ry_scaled)

        print(f"\nFinal domain radii (with LDomain_in_LSH='yes'):")
        for i, (rx, ry) in enumerate(zip(domain_rx_scaled, domain_ry_scaled)):
            domain_type = "main" if i < len(m['DomainType']) else "ABC"
            print(f"  Domain {i + 1} ({domain_type}): Rx = {rx:.3f}, Ry = {ry:.3f}")

    print(f"{'=' * 60}\n")


def build_cylindrical_for_frequency(model_dict, frequency=None):
    """
    Build a 2-D cylindrical mesh optimized for a specific frequency.
    If frequency is None, uses the maximum frequency from the range.
    """
    gmsh.initialize()
    gmsh.model.add("WaveGuide")

    m = model_dict['Model']

    # Calculate minimum velocity for mesh sizing
    min_velocity = get_min_velocity(model_dict)

    # Determine target frequency for mesh sizing
    if frequency is None:
        # Use maximum frequency for conservative mesh
        frequency = m['f_array_range']['end']

    # DEBUG: Show domain calculation details
    debug_domain_calculation(model_dict, frequency, min_velocity)

    # Calculate mesh sizes based on wavelength at target frequency
    wavelength = min_velocity / (frequency * 1000)  # frequency in kHz to Hz

    hmax = wavelength / 6  # Maximum element size = λ/6
    hmin = wavelength / 8  # Minimum element size = λ/8

    print(f"Mesh for {frequency} kHz: hmin={hmin:.6f}m, hmax={hmax:.6f}m "
          f"(based on min velocity {min_velocity:.0f} m/s)")

    # Create copies to avoid modifying original data
    domain_rx = m['DomainRx'].copy()
    domain_ry = m['DomainRy'].copy()
    domain_theta = m['DomainTheta'].copy()
    domain_ecc = m['DomainEcc'].copy()
    domain_ecc_angle = m['DomainEccAngle'].copy()
    domain_param = m['DomainParam'].copy()
    domain_type = m['DomainType'].copy()

    # Apply MATLAB-like logic for domain scaling based on LDomain_in_LSH
    ldomain_in_lsh = m.get('LDomain_in_LSH', 'none').lower()
    add_domain_exist = 'AddDomainType' in m and m['AddDomainType'].lower() != 'none'

    if ldomain_in_lsh == 'yes' and add_domain_exist:
        # Scale domains based on wavelength (MATLAB logic)
        if m['AddDomainLoc'].lower() == 'ext':
            # Scale the last main domain
            domain_rx[-1] = domain_rx[-2] + domain_rx[-1] * wavelength
            domain_ry[-1] = domain_ry[-2] + domain_ry[-1] * wavelength

            # Extend arrays with last domain parameters
            domain_theta = np.append(domain_theta, domain_theta[-1])
            domain_ecc = np.append(domain_ecc, domain_ecc[-1])
            domain_ecc_angle = np.append(domain_ecc_angle, domain_ecc_angle[-1])
            domain_param.append(domain_param[-1])
            domain_type.append(domain_type[-1])

            # Add ABC domain with scaled length
            abc_rx = domain_rx[-1] + m['AddDomainL'] * wavelength
            abc_ry = domain_ry[-1] + m['AddDomainL'] * wavelength
            domain_rx = np.append(domain_rx, abc_rx)
            domain_ry = np.append(domain_ry, abc_ry)

            print(f"Applied wavelength scaling: ABC Rx={abc_rx:.3f}, Ry={abc_ry:.3f}")

    elif add_domain_exist:
        # Fixed domain sizes (current logic)
        if m['AddDomainLoc'].lower() == 'ext':
            # Extend arrays with last domain parameters
            domain_theta = np.append(domain_theta, domain_theta[-1])
            domain_ecc = np.append(domain_ecc, domain_ecc[-1])
            domain_ecc_angle = np.append(domain_ecc_angle, domain_ecc_angle[-1])
            domain_param.append(domain_param[-1])
            domain_type.append(domain_type[-1])

            # Extend radii for ABC domain with fixed size
            abc_rx = domain_rx[-1] + m['AddDomainL']
            abc_ry = domain_ry[-1] + m['AddDomainL']
            domain_rx = np.append(domain_rx, abc_rx)
            domain_ry = np.append(domain_ry, abc_ry)

            print(f"Used fixed domain sizes: ABC Rx={abc_rx:.3f}, Ry={abc_ry:.3f}")

    layers = len(domain_type)

    print(f"Creating {layers} domains for frequency {frequency} kHz:")
    for i in range(layers):
        print(f"  Domain {i + 1}: Rx={domain_rx[i]:.3f}, Ry={domain_ry[i]:.3f}, Type={domain_type[i]}")

    # ---------- create all domains ----------
    surf_tags = []

    for i in range(layers):
        rx = domain_rx[i]
        ry = domain_ry[i]
        ecc = domain_ecc[i]
        ang = domain_ecc_angle[i] * np.pi / 180
        rot = domain_theta[i] * np.pi / 180

        # Create ellipse/disk
        tag = gmsh.model.occ.addDisk(0, 0, 0, rx, ry)

        # Apply transformations
        if ecc != 0:
            dx = ecc * np.cos(ang)
            dy = ecc * np.sin(ang)
            gmsh.model.occ.translate([(2, tag)], dx, dy, 0)

        if rot != 0:
            gmsh.model.occ.rotate([(2, tag)], 0, 0, 0, 0, 0, 1, rot)

        surf_tags.append(tag)

    gmsh.model.occ.synchronize()

    # ---------- boolean operations for nested domains ----------
    # Sort surfaces by area (largest first)
    areas = []
    for tag in surf_tags:
        mass_props = gmsh.model.occ.getMass(2, tag)
        areas.append(mass_props)

    sorted_tags = [tag for _, tag in sorted(zip(areas, surf_tags), reverse=True)]

    # Perform boolean operations to create nested structure
    for i in range(len(sorted_tags) - 1):
        result = gmsh.model.occ.cut([(2, sorted_tags[i])], [(2, sorted_tags[i + 1])],
                                    removeObject=True, removeTool=False)
        gmsh.model.occ.synchronize()

        if result[0]:
            sorted_tags[i] = result[0][0][1]

    gmsh.model.occ.synchronize()

    # ---------- physical groups ----------
    all_surfaces = gmsh.model.getEntities(2)

    # Sort surfaces by distance from center
    centers = []
    for dim_tag in all_surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim_tag[0], dim_tag[1])
        distance = np.sqrt(com[0] ** 2 + com[1] ** 2)
        centers.append(distance)

    sorted_final = [surf for _, surf in sorted(zip(centers, all_surfaces))]

    for i, (dim, tag) in enumerate(sorted_final):
        gmsh.model.addPhysicalGroup(2, [tag], i + 1)
        if i < len(domain_type):
            domain_name = f"layer_{i + 1}_{domain_type[i]}"
        else:
            domain_name = f"abc_domain"
        gmsh.model.setPhysicalName(2, i + 1, domain_name)

    # ---------- meshing with controlled element sizes ----------
    # Set mesh size constraints based on frequency
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmin)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)

    # Set gradient control (10% maximum size change between adjacent elements)
    gmsh.option.setNumber("Mesh.SmoothRatio", 1.1)

    # Use Frontal-Delaunay algorithm for better quality
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # High-order elements (Tri6)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    node_tags, coord, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    gmsh.finalize()

    return MeshContainer(node_tags, coord, elem_types, elem_tags, elem_node_tags)