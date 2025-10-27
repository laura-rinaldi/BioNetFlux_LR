
import sys
import os

# Add the path to folder B
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ooc1d', 'geometry')))


from domain_geometry import DomainGeometry

# Create geometry for Y-junction network
geometry = DomainGeometry(name="y_junction_network")

# Add main vessel (horizontal segment)
main_id = geometry.add_domain(
    extrema_start=(0.0, 0.0),
    extrema_end=(2.0, 0.0),
    name="main_vessel",
    vessel_type="parent",
    diameter=1.0
)

# Add upper branch
upper_id = geometry.add_domain(
    extrema_start=(2.0, 0.0),
    extrema_end=(3.0, 1.0),
    name="upper_branch",
    vessel_type="daughter",
    diameter=0.7
)

# Add lower branch  
lower_id = geometry.add_domain(
    extrema_start=(2.0, 0.0),
    extrema_end=(3.0, -1.0),
    name="lower_branch",
    vessel_type="daughter",
    diameter=0.7
)

# Set global properties
geometry.set_global_metadata(
    fluid_type="blood",
    viscosity=0.004,  # Pa·s
    density=1060,     # kg/m³
    problem_type="organ_on_chip"
)

# Analyze geometry
print(geometry.summary())
print(f"\nTotal network length: {geometry.total_length():.3f}")

# Access individual domains
for domain in geometry:
    center = domain.center_point()
    direction = domain.direction_vector()
    print(f"Domain {domain.name}:")
    print(f"  Center: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"  Direction: ({direction[0]:.2f}, {direction[1]:.2f})")
    
    # Access metadata
    if "diameter" in domain.metadata:
        print(f"  Diameter: {domain.metadata['diameter']}")

# Find specific domain
main_domain_id = geometry.find_domain_by_name("main_vessel")
if main_domain_id is not None:
    main_domain = geometry[main_domain_id]
    print(f"\nMain vessel length: {main_domain.euclidean_length():.3f}")
