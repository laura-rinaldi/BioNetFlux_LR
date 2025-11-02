#!/usr/bin/env python3
"""
Comprehensive test script for the DomainGeometry module.

This script tests all functionality of the geometry module including:
- Basic domain operations
- Geometry validation
- Connectivity analysis
- Parameter space management
- Error handling
- Performance with large geometries

Usage:
    python test_geometry.py
"""


import sys
import os
import numpy as np
import time
from typing import Dict, Any

# Add the code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.geometry import DomainGeometry, DomainInfo

def run_basic_functionality_tests() -> bool:
    """Test basic geometry operations."""
    print("Testing Basic Functionality")
    print("-" * 40)
    
    all_passed = True
    
    # Test 1: Empty geometry
    print("Test 1: Empty geometry creation")
    try:
        empty_geom = DomainGeometry("empty_test")
        assert len(empty_geom) == 0
        assert empty_geom.num_domains() == 0
        assert empty_geom.name == "empty_test"
        print("  ✓ Empty geometry creation")
    except Exception as e:
        print(f"  ✗ Empty geometry creation failed: {e}")
        all_passed = False
    
    # Test 2: Domain addition
    print("Test 2: Domain addition")
    try:
        geom = DomainGeometry("test_geom")
        domain_id = geom.add_domain((0.0, 0.0), (1.0, 0.0), name="test_domain")
        assert domain_id == 0
        assert len(geom) == 1
        assert geom.get_domain(0).name == "test_domain"
        print("  ✓ Domain addition")
    except Exception as e:
        print(f"  ✗ Domain addition failed: {e}")
        all_passed = False
    
    # Test 3: Domain retrieval
    print("Test 3: Domain retrieval and properties")
    try:
        domain = geom.get_domain(0)
        assert domain.extrema_start == (0.0, 0.0)
        assert domain.extrema_end == (1.0, 0.0)
        assert abs(domain.euclidean_length() - 1.0) < 1e-12
        assert domain.center_point() == (0.5, 0.0)
        assert domain.direction_vector() == (1.0, 0.0)
        print("  ✓ Domain retrieval and properties")
    except Exception as e:
        print(f"  ✗ Domain retrieval failed: {e}")
        all_passed = False
    
    # Test 4: Multiple domains
    print("Test 4: Multiple domain operations")
    try:
        geom.add_domain((1.0, 0.0), (1.0, 1.0), name="domain2")
        geom.add_domain((1.0, 1.0), (0.0, 1.0), name="domain3")
        
        assert len(geom) == 3
        assert len(geom.get_domain_names()) == 3
        assert geom.find_domain_by_name("domain2") == 1
        assert geom.find_domain_by_name("nonexistent") is None
        print("  ✓ Multiple domain operations")
    except Exception as e:
        print(f"  ✗ Multiple domain operations failed: {e}")
        all_passed = False
    
    # Test 5: Bounding box
    print("Test 5: Bounding box calculation")
    try:
        bbox = geom.get_bounding_box()
        expected = {'x_min': 0.0, 'x_max': 1.0, 'y_min': 0.0, 'y_max': 1.0}
        for key, expected_val in expected.items():
            assert abs(bbox[key] - expected_val) < 1e-12, f"Bounding box {key} mismatch"
        print("  ✓ Bounding box calculation")
    except Exception as e:
        print(f"  ✗ Bounding box calculation failed: {e}")
        all_passed = False
    
    return all_passed

def run_validation_tests() -> bool:
    """Test geometry validation functionality."""
    print("\nTesting Validation Functionality")
    print("-" * 40)
    
    all_passed = True
    
    # Test valid geometry
    print("Test 1: Valid geometry validation")
    try:
        valid_geom = DomainGeometry("valid_test")
        valid_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="seg1")
        valid_geom.add_domain((1.0, 0.0), (2.0, 0.0), name="seg2")
        
        is_valid = valid_geom.validate_geometry(verbose=True)
        assert is_valid == True
        print("  ✓ Valid geometry validation")
    except Exception as e:
        print(f"  ✗ Valid geometry validation failed: {e}")
        all_passed = False
    
    # Test invalid geometries
    print("Test 2: Invalid geometry detection")
    try:
        # Zero-length domain
        invalid_geom = DomainGeometry("invalid_test")
        invalid_geom.add_domain((0.0, 0.0), (0.0, 0.0), name="zero_length")
        
        is_valid = invalid_geom.validate_geometry(verbose=False)
        assert is_valid == False
        print("  ✓ Zero-length domain detected")
    except Exception as e:
        print(f"  ✗ Invalid geometry detection failed: {e}")
        all_passed = False
    
    # Test duplicate names
    print("Test 3: Duplicate name detection")
    try:
        dup_geom = DomainGeometry("duplicate_test")
        dup_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="duplicate")
        dup_geom.add_domain((2.0, 0.0), (3.0, 0.0), name="duplicate")
        
        is_valid = dup_geom.validate_geometry(verbose=False)
        assert is_valid == False
        print("  ✓ Duplicate names detected")
    except Exception as e:
        print(f"  ✗ Duplicate name detection failed: {e}")
        all_passed = False
    
    # Test overlapping parameter spaces (should be warning, not error)
    print("Test 4: Overlapping parameter spaces (warning test)")
    try:
        overlap_geom = DomainGeometry("overlap_test")
        overlap_geom.add_domain((0.0, 0.0), (1.0, 0.0), 
                               domain_start=0.0, domain_length=1.5, name="domain1")
        overlap_geom.add_domain((1.0, 0.0), (2.0, 0.0), 
                               domain_start=1.0, domain_length=1.5, name="domain2")
        
        is_valid = overlap_geom.validate_geometry(verbose=True)
        # Should still be valid (overlapping parameters are just a warning)
        assert is_valid == True
        print("  ✓ Overlapping parameter spaces produce warning, not error")
    except Exception as e:
        print(f"  ✗ Overlapping parameter space test failed: {e}")
        all_passed = False
    
    return all_passed

def run_connectivity_tests() -> bool:
    """Test connectivity analysis functionality."""
    print("\nTesting Connectivity Analysis")
    print("-" * 40)
    
    all_passed = True
    
    # Test connected geometry
    print("Test 1: Connected geometry analysis")
    try:
        connected_geom = DomainGeometry("connected_test")
        connected_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="seg1")
        connected_geom.add_domain((1.0, 0.0), (2.0, 0.0), name="seg2")
        connected_geom.add_domain((2.0, 0.0), (2.0, 1.0), name="seg3")
        
        connectivity = connected_geom.get_connectivity_info()
        assert connectivity['is_connected'] == True
        assert connectivity['num_components'] == 1
        assert len(connectivity['intersections']) >= 2
        print("  ✓ Connected geometry analysis")
    except Exception as e:
        print(f"  ✗ Connected geometry analysis failed: {e}")
        all_passed = False
    
    # Test disconnected geometry
    print("Test 2: Disconnected geometry analysis")
    try:
        disconnected_geom = DomainGeometry("disconnected_test")
        disconnected_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="island1")
        disconnected_geom.add_domain((3.0, 3.0), (4.0, 3.0), name="island2")
        
        connectivity = disconnected_geom.get_connectivity_info()
        assert connectivity['is_connected'] == False
        assert connectivity['num_components'] == 2
        assert len(connectivity['isolated_domains']) == 2
        print("  ✓ Disconnected geometry analysis")
    except Exception as e:
        print(f"  ✗ Disconnected geometry analysis failed: {e}")
        all_passed = False
    
    # Test intersection finding
    print("Test 3: Intersection detection")
    try:
        t_geom = DomainGeometry("t_junction_test")
        t_geom.add_domain((0.0, -1.0), (0.0, 1.0), name="vertical")
        t_geom.add_domain((-1.0, 0.0), (1.0, 0.0), name="horizontal")
        
        intersections = t_geom.find_intersections(tolerance=1e-6)
        # Should find intersection at (0, 0)
        assert len(intersections) >= 1
        print("  ✓ Intersection detection")
    except Exception as e:
        print(f"  ✗ Intersection detection failed: {e}")
        all_passed = False
    
    return all_passed

def run_parameter_space_tests() -> bool:
    """Test parameter space management."""
    print("\nTesting Parameter Space Management")
    print("-" * 40)
    
    all_passed = True
    
    # Test parameter space suggestions
    print("Test 1: Parameter space suggestions")
    try:
        geom = DomainGeometry("param_test")
        geom.add_domain((0.0, 0.0), (1.0, 0.0), domain_start=0.0, domain_length=1.0)
        geom.add_domain((1.0, 0.0), (3.0, 0.0), domain_start=0.5, domain_length=2.0)  # Overlapping
        
        suggestions = geom.suggest_parameter_spacing(gap=0.1)
        assert len(suggestions) == 2
        
        # Check that suggestions don't overlap
        start1, len1 = suggestions[0]
        start2, len2 = suggestions[1]
        assert start2 >= start1 + len1 + 0.1  # Should have gap
        print("  ✓ Parameter space suggestions")
    except Exception as e:
        print(f"  ✗ Parameter space suggestions failed: {e}")
        all_passed = False
    
    # Test custom parameter spaces
    print("Test 2: Custom parameter spaces")
    try:
        custom_geom = DomainGeometry("custom_param_test")
        custom_geom.add_domain((0.0, 0.0), (1.0, 0.0), 
                              domain_start=10.0, domain_length=5.0, name="custom1")
        custom_geom.add_domain((1.0, 0.0), (2.0, 0.0), 
                              domain_start=20.0, domain_length=3.0, name="custom2")
        
        domain1 = custom_geom.get_domain(0)
        domain2 = custom_geom.get_domain(1)
        
        assert domain1.domain_start == 10.0
        assert domain1.domain_length == 5.0
        assert domain2.domain_start == 20.0
        assert domain2.domain_length == 3.0
        print("  ✓ Custom parameter spaces")
    except Exception as e:
        print(f"  ✗ Custom parameter spaces failed: {e}")
        all_passed = False
    
    return all_passed

def run_error_handling_tests() -> bool:
    """Test error handling and edge cases."""
    print("\nTesting Error Handling")
    print("-" * 40)
    
    all_passed = True
    
    # Test invalid domain access
    print("Test 1: Invalid domain access")
    try:
        geom = DomainGeometry("error_test")
        geom.add_domain((0.0, 0.0), (1.0, 0.0))
        
        # Should raise IndexError
        try:
            invalid_domain = geom.get_domain(5)
            print("  ✗ Should have raised IndexError")
            all_passed = False
        except IndexError:
            print("  ✓ IndexError properly raised")
        except Exception as e:
            print(f"  ✗ Wrong exception type: {e}")
            all_passed = False
            
    except Exception as e:
        print(f"  ✗ Error handling test setup failed: {e}")
        all_passed = False
    
    # Test domain removal
    print("Test 2: Domain removal")
    try:
        geom = DomainGeometry("removal_test")
        id1 = geom.add_domain((0.0, 0.0), (1.0, 0.0), name="remove_me")
        id2 = geom.add_domain((1.0, 0.0), (2.0, 0.0), name="keep_me")
        
        assert len(geom) == 2
        geom.remove_domain(id1)
        assert len(geom) == 1
        assert geom.get_domain(0).name == "keep_me"
        print("  ✓ Domain removal")
    except Exception as e:
        print(f"  ✗ Domain removal failed: {e}")
        all_passed = False
    
    # Test empty geometry operations
    print("Test 3: Empty geometry operations")
    try:
        empty_geom = DomainGeometry("empty_ops_test")
        
        # These should not crash
        bbox = empty_geom.get_bounding_box()
        connectivity = empty_geom.get_connectivity_info()
        suggestions = empty_geom.suggest_parameter_spacing()
        
        assert len(suggestions) == 0
        assert connectivity['num_components'] == 0
        print("  ✓ Empty geometry operations")
    except Exception as e:
        print(f"  ✗ Empty geometry operations failed: {e}")
        all_passed = False
    
    return all_passed

def run_test_geometries() -> bool:
    """Test the predefined test geometries."""
    print("\nTesting Predefined Test Geometries")
    print("-" * 40)
    
    all_passed = True
    
    try:
        test_geometries = DomainGeometry.create_test_geometries()
        
        expected_geometries = ["linear", "t_junction", "grid", "star", "branching", "degenerate"]
        
        for name in expected_geometries:
            if name not in test_geometries:
                print(f"  ✗ Missing test geometry: {name}")
                all_passed = False
                continue
            
            geom = test_geometries[name]
            
            # Run self-test on each geometry
            if name != "degenerate":  # Degenerate geometry is expected to fail validation
                test_result = geom.run_self_test(verbose=False)
                if test_result:
                    print(f"  ✓ {name} geometry self-test passed")
                else:
                    print(f"  ✗ {name} geometry self-test failed")
                    all_passed = False
            else:
                # Degenerate case should fail validation
                is_valid = geom.validate_geometry(verbose=False)
                if not is_valid:
                    print(f"  ✓ {name} geometry correctly identified as invalid")
                else:
                    print(f"  ✗ {name} geometry should be invalid")
                    all_passed = False
        
    except Exception as e:
        print(f"  ✗ Test geometries creation failed: {e}")
        all_passed = False
    
    return all_passed

def run_performance_tests() -> bool:
    """Test performance with larger geometries."""
    print("\nTesting Performance")
    print("-" * 40)
    
    all_passed = True
    
    # Test large geometry creation
    print("Test 1: Large geometry creation")
    try:
        start_time = time.time()
        
        large_geom = DomainGeometry("large_test")
        n_domains = 1000
        
        for i in range(n_domains):
            large_geom.add_domain((i, 0), (i+1, 0), name=f"domain_{i}")
        
        creation_time = time.time() - start_time
        
        assert len(large_geom) == n_domains
        print(f"  ✓ Created {n_domains} domains in {creation_time:.3f} seconds")
        
        if creation_time > 5.0:  # Should be much faster than this
            print(f"  ⚠ Warning: Creation time seems slow ({creation_time:.3f}s)")
    
    except Exception as e:
        print(f"  ✗ Large geometry creation failed: {e}")
        all_passed = False
    
    # Test large geometry operations
    print("Test 2: Large geometry operations")
    try:
        start_time = time.time()
        
        # Test validation
        large_geom.validate_geometry(verbose=False)
        
        # Test bounding box
        bbox = large_geom.get_bounding_box()
        
        # Test connectivity (this might be slow for very large geometries)
        connectivity = large_geom.get_connectivity_info()
        
        operation_time = time.time() - start_time
        print(f"  ✓ Large geometry operations completed in {operation_time:.3f} seconds")
        
        if operation_time > 10.0:  # Adjust threshold as needed
            print(f"  ⚠ Warning: Operations seem slow ({operation_time:.3f}s)")
    
    except Exception as e:
        print(f"  ✗ Large geometry operations failed: {e}")
        all_passed = False
    
    return all_passed

def main():
    """Run all geometry tests."""
    print("="*60)
    print("DOMAIN GEOMETRY MODULE COMPREHENSIVE TEST")
    print("="*60)
    
    test_functions = [
        run_basic_functionality_tests,
        run_validation_tests,
        run_connectivity_tests,
        run_parameter_space_tests,
        run_error_handling_tests,
        run_test_geometries,
        run_performance_tests
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test function {test_func.__name__} crashed: {e}")
            results.append(False)
    
    total_time = time.time() - total_start_time
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {total_time:.3f} seconds")
    
    if passed == total:
        print("✓ ALL TESTS PASSED!")
        return_code = 0
    else:
        print("✗ SOME TESTS FAILED!")
        return_code = 1
    
    # Demonstrate usage example
    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)
    
    try:
        # Create and test a simple geometry
        example_geom = DomainGeometry("example_usage")
        example_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="segment1")
        example_geom.add_domain((1.0, 0.0), (1.0, 1.0), name="segment2")
        
        print("Created example geometry:")
        print(example_geom.summary())
        
        print("\nRunning self-test:")
        example_geom.run_self_test(verbose=True)
        
    except Exception as e:
        print(f"Usage example failed: {e}")
    
    return return_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
