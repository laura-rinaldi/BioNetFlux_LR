import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DomainInfo:
    """
    Container for domain geometric information.
    """
    domain_id: int
    extrema_start: Tuple[float, float]  # (x1, y1)
    extrema_end: Tuple[float, float]    # (x2, y2)
    domain_start: float = 0.0           # Parameter space start
    domain_length: float = 1.0          # Parameter space length
    name: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate default domain_length from extrema if not set
        if self.domain_length == 1.0:  # Default value
            self.domain_length = self.euclidean_length()
    
    def euclidean_length(self) -> float:
        """Calculate Euclidean distance between extrema."""
        dx = self.extrema_end[0] - self.extrema_start[0]
        dy = self.extrema_end[1] - self.extrema_start[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def center_point(self) -> Tuple[float, float]:
        """Calculate center point between extrema."""
        x_center = (self.extrema_start[0] + self.extrema_end[0]) / 2
        y_center = (self.extrema_start[1] + self.extrema_end[1]) / 2
        return (x_center, y_center)
    
    def direction_vector(self) -> Tuple[float, float]:
        """Calculate unit direction vector from start to end."""
        dx = self.extrema_end[0] - self.extrema_start[0]
        dy = self.extrema_end[1] - self.extrema_start[1]
        length = self.euclidean_length()
        if length > 0:
            return (dx/length, dy/length)
        return (1.0, 0.0)


class DomainGeometry:
    """
    Lean geometry class for constructing and handling complex multi-domain geometries.
    
    Manages a collection of domains (segments) and provides interface methods 
    for problem and discretization setup.
    """
    
    def __init__(self, name: str = "unnamed_geometry"):
        """
        Initialize empty geometry.
        
        Args:
            name: Descriptive name for the geometry
        """
        self.name = name
        self.domains: List[DomainInfo] = []
        self._next_id = 0
        self._global_metadata: Dict[str, Any] = {}
    
    def add_domain(self, 
                   extrema_start: Tuple[float, float],
                   extrema_end: Tuple[float, float],
                   domain_start: Optional[float] = None,
                   domain_length: Optional[float] = None,
                   name: Optional[str] = None,
                   **metadata) -> int:
        """
        Add a domain (segment) to the geometry.
        
        Args:
            extrema_start: Start point (x1, y1) in physical space
            extrema_end: End point (x2, y2) in physical space
            domain_start: Parameter space start (default: 0.0)
            domain_length: Parameter space length (default: Euclidean distance)
            name: Optional domain name
            **metadata: Additional domain-specific metadata
            
        Returns:
            Domain ID (index) of the added domain
        """
        # Calculate default values
        if domain_start is None:
            domain_start = 0.0
        
        if domain_length is None:
            # Default to Euclidean distance between extrema
            dx = extrema_end[0] - extrema_start[0]
            dy = extrema_end[1] - extrema_start[1]
            domain_length = np.sqrt(dx*dx + dy*dy)
        
        # Generate default name if not provided
        if name is None:
            name = f"domain_{self._next_id}"
        
        # Create domain info
        domain_info = DomainInfo(
            domain_id=self._next_id,
            extrema_start=extrema_start,
            extrema_end=extrema_end,
            domain_start=domain_start,
            domain_length=domain_length,
            name=name,
            metadata=metadata
        )
        
        # Add to collection
        self.domains.append(domain_info)
        domain_id = self._next_id
        self._next_id += 1
        
        return domain_id
    
    def get_domain(self, domain_id: int) -> DomainInfo:
        """
        Retrieve domain information by ID.
        
        Args:
            domain_id: Domain index
            
        Returns:
            DomainInfo object containing all domain parameters
            
        Raises:
            IndexError: If domain_id is invalid
        """
        if domain_id < 0 or domain_id >= len(self.domains):
            raise IndexError(f"Domain ID {domain_id} out of range [0, {len(self.domains)-1}]")
        
        return self.domains[domain_id]
    
    def get_all_domains(self) -> List[DomainInfo]:
        """Get list of all domains."""
        return self.domains.copy()
    
    def num_domains(self) -> int:
        """Get number of domains in geometry."""
        return len(self.domains)
    
    def get_bounding_box(self) -> Dict[str, float]:
        """
        Calculate bounding box of entire geometry.
        
        Returns:
            Dictionary with keys: x_min, x_max, y_min, y_max
        """
        if not self.domains:
            return {'x_min': 0.0, 'x_max': 1.0, 'y_min': 0.0, 'y_max': 1.0}
        
        all_x = []
        all_y = []
        
        for domain in self.domains:
            all_x.extend([domain.extrema_start[0], domain.extrema_end[0]])
            all_y.extend([domain.extrema_start[1], domain.extrema_end[1]])
        
        return {
            'x_min': min(all_x), 
            'x_max': max(all_x),
            'y_min': min(all_y),
            'y_max': max(all_y)
        }
    
    def set_global_metadata(self, **metadata):
        """Set global metadata for the entire geometry."""
        self._global_metadata.update(metadata)
    
    def get_global_metadata(self) -> Dict[str, Any]:
        """Get global metadata."""
        return self._global_metadata.copy()
    
    def get_domain_names(self) -> List[str]:
        """Get list of all domain names."""
        return [domain.name for domain in self.domains]
    
    def find_domain_by_name(self, name: str) -> Optional[int]:
        """
        Find domain ID by name.
        
        Args:
            name: Domain name to search for
            
        Returns:
            Domain ID if found, None otherwise
        """
        for domain in self.domains:
            if domain.name == name:
                return domain.domain_id
        return None
    
    def remove_domain(self, domain_id: int):
        """
        Remove domain by ID.
        
        Args:
            domain_id: Domain ID to remove
            
        Raises:
            IndexError: If domain_id is invalid
        """
        if domain_id < 0 or domain_id >= len(self.domains):
            raise IndexError(f"Domain ID {domain_id} out of range")
        
        # Remove domain and update IDs
        del self.domains[domain_id]
        
        # Renumber domain IDs to maintain consistency
        for i, domain in enumerate(self.domains):
            domain.domain_id = i
        
        self._next_id = len(self.domains)
    
    def total_length(self) -> float:
        """Calculate total Euclidean length of all domains."""
        return sum(domain.euclidean_length() for domain in self.domains)
    
    def summary(self) -> str:
        """Generate summary string of the geometry."""
        lines = [
            f"Geometry: {self.name}",
            f"Number of domains: {len(self.domains)}",
            f"Total length: {self.total_length():.3f}",
            "Domains:"
        ]
        
        for domain in self.domains:
            lines.append(f"  {domain.domain_id}: {domain.name}")
            lines.append(f"    Extrema: {domain.extrema_start} → {domain.extrema_end}")
            lines.append(f"    Parameter: [{domain.domain_start:.3f}, {domain.domain_start + domain.domain_length:.3f}]")
            lines.append(f"    Length: {domain.euclidean_length():.3f}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        """Support len() operation."""
        return len(self.domains)
    
    def __getitem__(self, domain_id: int) -> DomainInfo:
        """Support indexing: geometry[i]."""
        return self.get_domain(domain_id)
    
    def __iter__(self):
        """Support iteration over domains."""
        return iter(self.domains)
    
    def validate_geometry(self, verbose: bool = False) -> bool:
        """
        Validate the geometry for consistency and common issues.
        
        Args:
            verbose: Whether to print validation details
            
        Returns:
            True if geometry is valid, False otherwise
        """
        if verbose:
            print(f"Validating geometry: {self.name}")
        
        issues = []
        warnings = []
        
        # Check if geometry is empty
        if len(self.domains) == 0:
            issues.append("Geometry is empty (no domains)")
        
        # Validate individual domains
        for domain in self.domains:
            # Check for degenerate domains (zero length)
            if domain.euclidean_length() < 1e-12:
                issues.append(f"Domain {domain.domain_id} ({domain.name}) has zero length")
            
            # Check for negative parameter space length
            if domain.domain_length <= 0:
                issues.append(f"Domain {domain.domain_id} ({domain.name}) has non-positive parameter length")
            
            # Check for valid coordinates
            for coord_name, coord in [("start", domain.extrema_start), ("end", domain.extrema_end)]:
                if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
                    issues.append(f"Domain {domain.domain_id} ({domain.name}) has invalid {coord_name} coordinates")
                elif not all(isinstance(x, (int, float)) for x in coord):
                    issues.append(f"Domain {domain.domain_id} ({domain.name}) has non-numeric {coord_name} coordinates")
        
        # Check for duplicate domain names
        names = [domain.name for domain in self.domains if domain.name is not None]
        if len(names) != len(set(names)):
            duplicate_names = [name for name in set(names) if names.count(name) > 1]
            issues.append(f"Duplicate domain names found: {duplicate_names}")
        
        # Check for overlapping parameter spaces (WARNING, not failure)
        param_ranges = [(domain.domain_start, domain.domain_start + domain.domain_length) 
                       for domain in self.domains]
        for i, (start1, end1) in enumerate(param_ranges):
            for j, (start2, end2) in enumerate(param_ranges[i+1:], i+1):
                if not (end1 <= start2 or end2 <= start1):  # Overlapping ranges
                    domain1 = self.domains[i]
                    domain2 = self.domains[j]
                    warnings.append(f"Overlapping parameter spaces: {domain1.name} [{start1:.3f}, {end1:.3f}] "
                                  f"and {domain2.name} [{start2:.3f}, {end2:.3f}]")
        
        # Report results
        if verbose:
            if issues:
                print(f"  Found {len(issues)} validation errors:")
                for issue in issues:
                    print(f"    ✗ {issue}")
            else:
                print("  ✓ No validation errors found")
            
            if warnings:
                print(f"  Found {len(warnings)} warnings:")
                for warning in warnings:
                    print(f"    ⚠ {warning}")
            else:
                print("  ✓ No warnings")
        
        # Return True only if no critical issues (warnings don't affect validation)
        return len(issues) == 0
    
    def find_intersections(self, tolerance: float = 1e-6) -> List[Tuple[int, int, Tuple[float, float]]]:
        """
        Find intersection points between domain segments.
        
        This method finds both endpoint connections and actual line segment intersections.
        
        Args:
            tolerance: Distance tolerance for considering points as intersecting
            
        Returns:
            List of tuples (domain1_id, domain2_id, intersection_point)
        """
        intersections = []
        
        def segments_intersect(p1, q1, p2, q2, tol=tolerance):
            """
            Find intersection point between two line segments.
            
            Args:
                p1, q1: Start and end points of first segment
                p2, q2: Start and end points of second segment
                tol: Tolerance for numerical precision
                
            Returns:
                Intersection point (x, y) if segments intersect, None otherwise
            """
            x1, y1 = p1
            x2, y2 = q1
            x3, y3 = p2
            x4, y4 = q2
            
            # Calculate direction vectors
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x4 - x3, y4 - y3
            
            # Calculate determinant
            det = dx1 * dy2 - dy1 * dx2
            
            # Check if lines are parallel
            if abs(det) < tol:
                # Lines are parallel, check if they're collinear and overlapping
                # Vector from p1 to p2
                dx3, dy3 = x3 - x1, y3 - y1
                
                # Check if points are collinear using cross product
                cross = dx3 * dy1 - dy3 * dx1
                if abs(cross) < tol:
                    # Segments are collinear, check for overlap
                    # Project all points onto the line direction
                    if abs(dx1) > abs(dy1):  # Project onto x-axis
                        t1_start = 0.0
                        t1_end = 1.0
                        t2_start = (x3 - x1) / dx1 if abs(dx1) > tol else 0.0
                        t2_end = (x4 - x1) / dx1 if abs(dx1) > tol else 0.0
                    else:  # Project onto y-axis
                        t1_start = 0.0
                        t1_end = 1.0
                        t2_start = (y3 - y1) / dy1 if abs(dy1) > tol else 0.0
                        t2_end = (y4 - y1) / dy1 if abs(dy1) > tol else 0.0
                    
                    # Ensure t2_start <= t2_end
                    if t2_start > t2_end:
                        t2_start, t2_end = t2_end, t2_start
                    
                    # Check for overlap
                    overlap_start = max(t1_start, t2_start)
                    overlap_end = min(t1_end, t2_end)
                    
                    if overlap_start <= overlap_end + tol:
                        # Segments overlap, return midpoint of overlap
                        t_mid = (overlap_start + overlap_end) / 2
                        intersection_x = x1 + t_mid * dx1
                        intersection_y = y1 + t_mid * dy1
                        return (intersection_x, intersection_y)
                
                return None  # Parallel but not collinear, or no overlap
            
            # Calculate intersection parameters
            dx3, dy3 = x1 - x3, y1 - y3
            t = (dx2 * dy3 - dy2 * dx3) / det
            u = (dx1 * dy3 - dy1 * dx3) / det
            
            # Check if intersection point lies on both segments
            if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
                # Calculate intersection point
                intersection_x = x1 + t * dx1
                intersection_y = y1 + t * dy1
                return (intersection_x, intersection_y)
            
            return None
        
        def point_distance(p1, p2):
            """Calculate distance between two points."""
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Check all pairs of domains
        for i, domain1 in enumerate(self.domains):
            for j, domain2 in enumerate(self.domains[i+1:], i+1):
                # Get segment endpoints
                seg1_start = domain1.extrema_start
                seg1_end = domain1.extrema_end
                seg2_start = domain2.extrema_start  
                seg2_end = domain2.extrema_end
                
                # Find segment intersection
                intersection_point = segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end)
                
                if intersection_point is not None:
                    intersections.append((i, j, intersection_point))
                else:
                    # Also check for close endpoints (endpoint connections)
                    endpoints1 = [seg1_start, seg1_end]
                    endpoints2 = [seg2_start, seg2_end]
                    
                    for ep1 in endpoints1:
                        for ep2 in endpoints2:
                            if point_distance(ep1, ep2) <= tolerance:
                                # Use the midpoint as intersection
                                midpoint = ((ep1[0] + ep2[0]) / 2, (ep1[1] + ep2[1]) / 2)
                                intersections.append((i, j, midpoint))
                                break
                        else:
                            continue
                        break
        
        return intersections
    
    def get_connectivity_info(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Analyze geometry connectivity.
        
        Args:
            tolerance: Distance tolerance for considering points as connected
            
        Returns:
            Dictionary with connectivity information
        """
        intersections = self.find_intersections(tolerance)
        
        # Build adjacency information
        adjacency = {i: set() for i in range(len(self.domains))}
        for domain1_id, domain2_id, _ in intersections:
            adjacency[domain1_id].add(domain2_id)
            adjacency[domain2_id].add(domain1_id)
        
        # Find connected components
        visited = set()
        components = []
        
        for domain_id in range(len(self.domains)):
            if domain_id not in visited:
                component = []
                stack = [domain_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(adjacency[current] - visited)
                
                components.append(component)
        
        return {
            'intersections': intersections,
            'adjacency': adjacency,
            'connected_components': components,
            'num_components': len(components),
            'is_connected': len(components) == 1,
            'isolated_domains': [comp[0] for comp in components if len(comp) == 1]
        }
    
    def suggest_parameter_spacing(self, gap: float = 0.1) -> List[Tuple[float, float]]:
        """
        Suggest non-overlapping parameter space ranges for all domains.
        
        Args:
            gap: Minimum gap between parameter ranges
            
        Returns:
            List of (start, length) tuples for each domain
        """
        suggestions = []
        current_start = 0.0
        
        # Sort domains by current parameter start for consistent ordering
        sorted_domains = sorted(self.domains, key=lambda d: d.domain_start)
        
        for domain in sorted_domains:
            # Use the domain's preferred length, or Euclidean length as fallback
            length = domain.domain_length if domain.domain_length > 0 else domain.euclidean_length()
            
            suggestions.append((current_start, length))
            current_start += length + gap
        
        return suggestions
    
    @classmethod
    def create_test_geometries(cls) -> Dict[str, 'DomainGeometry']:
        """
        Create a collection of test geometries for validation and testing.
        
        Returns:
            Dictionary of test geometry instances
        """
        geometries = {}
        
        # 1. Simple linear chain
        linear = cls("linear_chain")
        linear.add_domain((0.0, 0.0), (1.0, 0.0), name="segment1")
        linear.add_domain((1.0, 0.0), (2.0, 0.0), name="segment2")
        linear.add_domain((2.0, 0.0), (3.0, 0.0), name="segment3")
        geometries["linear"] = linear
        
        # 2. T-junction
        t_junction = cls("t_junction")
        t_junction.add_domain((0.0, -1.0), (0.0, 1.0), name="main_channel")
        t_junction.add_domain((0.0, 0.0), (1.0, 0.0), name="side_branch")
        geometries["t_junction"] = t_junction
        
        # 3. Grid network
        grid = cls("grid_network")
        # Vertical segments
        grid.add_domain((-0.5, 0.0), (-0.5, 1.0), name="left_vertical")
        grid.add_domain((0.5, 0.0), (0.5, 1.0), name="right_vertical")
        # Horizontal connectors
        for i, y in enumerate([0.2, 0.4, 0.6, 0.8]):
            grid.add_domain((-0.5, y), (0.5, y), name=f"horizontal_{i+1}")
        geometries["grid"] = grid
        
        # 4. Star network
        star = cls("star_network")
        center = (0.0, 0.0)
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i, angle in enumerate(angles):
            end_point = (np.cos(angle), np.sin(angle))
            star.add_domain(center, end_point, name=f"branch_{i+1}")
        geometries["star"] = star
        
        # 5. Complex branching
        branching = cls("complex_branching")
        # Main trunk
        branching.add_domain((0.0, 0.0), (0.0, 2.0), name="trunk")
        # Primary branches
        branching.add_domain((0.0, 1.0), (1.0, 1.5), name="branch_1a")
        branching.add_domain((0.0, 1.0), (-1.0, 1.5), name="branch_1b")
        # Secondary branches
        branching.add_domain((1.0, 1.5), (1.5, 2.0), name="branch_2a")
        branching.add_domain((1.0, 1.5), (1.5, 1.0), name="branch_2b")
        geometries["branching"] = branching
        
        # 6. Degenerate case (for testing validation)
        degenerate = cls("degenerate_test")
        degenerate.add_domain((0.0, 0.0), (0.0, 0.0), name="zero_length")  # Zero length
        degenerate.add_domain((1.0, 1.0), (2.0, 1.0), domain_start=0.5, domain_length=1.0, name="overlap1")
        degenerate.add_domain((3.0, 1.0), (4.0, 1.0), domain_start=1.0, domain_length=1.0, name="overlap2")
        geometries["degenerate"] = degenerate
        
        return geometries
    
    def run_self_test(self, verbose: bool = True) -> bool:
        """
        Run comprehensive self-test on the geometry.
        
        Args:
            verbose: Whether to print detailed test results
            
        Returns:
            True if all tests pass, False otherwise
        """
        if verbose:
            print(f"Running self-test for geometry: {self.name}")
            print("=" * 50)
        
        all_passed = True
        
        # Test 1: Basic functionality
        if verbose:
            print("Test 1: Basic functionality")
        
        try:
            # Test domain addition and retrieval
            original_count = len(self.domains)
            test_id = self.add_domain((10.0, 10.0), (11.0, 11.0), name="test_domain")
            
            if len(self.domains) != original_count + 1:
                if verbose:
                    print("  ✗ Domain addition failed")
                all_passed = False
            elif self.get_domain(test_id).name != "test_domain":
                if verbose:
                    print("  ✗ Domain retrieval failed")
                all_passed = False
            else:
                if verbose:
                    print("  ✓ Domain addition and retrieval")
            
            # Clean up test domain
            self.remove_domain(test_id)
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Basic functionality test failed: {e}")
            all_passed = False
        
        # Test 2: Geometry validation
        if verbose:
            print("Test 2: Geometry validation")
        
        try:
            is_valid = self.validate_geometry(verbose=False)
            if verbose:
                print(f"  {'✓' if is_valid else '✗'} Geometry validation: {'PASS' if is_valid else 'FAIL'}")
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Validation test failed: {e}")
            all_passed = False
        
        # Test 3: Connectivity analysis
        if verbose:
            print("Test 3: Connectivity analysis")
        
        try:
            connectivity = self.get_connectivity_info()
            if verbose:
                print(f"  ✓ Found {len(connectivity['intersections'])} intersections")
                print(f"  ✓ {connectivity['num_components']} connected component(s)")
                if connectivity['isolated_domains']:
                    print(f"  ! {len(connectivity['isolated_domains'])} isolated domain(s)")
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Connectivity analysis failed: {e}")
            all_passed = False
        
        # Test 4: Bounding box calculation
        if verbose:
            print("Test 4: Bounding box calculation")
        
        try:
            bbox = self.get_bounding_box()
            required_keys = ['x_min', 'x_max', 'y_min', 'y_max']
            if all(key in bbox for key in required_keys):
                if verbose:
                    print(f"  ✓ Bounding box: x[{bbox['x_min']:.2f}, {bbox['x_max']:.2f}], y[{bbox['y_min']:.2f}, {bbox['y_max']:.2f}]")
            else:
                if verbose:
                    print("  ✗ Bounding box missing required keys")
                all_passed = False
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Bounding box calculation failed: {e}")
            all_passed = False
        
        # Test 5: Parameter space suggestions
        if verbose:
            print("Test 5: Parameter space analysis")
        
        try:
            suggestions = self.suggest_parameter_spacing()
            if len(suggestions) == len(self.domains):
                if verbose:
                    print("  ✓ Parameter space suggestions generated")
            else:
                if verbose:
                    print("  ✗ Parameter space suggestion count mismatch")
                all_passed = False
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Parameter space analysis failed: {e}")
            all_passed = False
        
        if verbose:
            print("=" * 50)
            print(f"Self-test result: {'PASS' if all_passed else 'FAIL'}")
        
        return all_passed
