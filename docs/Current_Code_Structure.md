# BioNetFlux Current Code Structure

*Generated documentation of the actual codebase structure and components*

## Directory Tree

```
BioNetFlux/
├── 📁 code/                           # Main source code directory
│   ├── 📁 ooc1d/                      # Core framework package
│   │   ├── 📄 __init__.py             # Package initialization
│   │   │
│   │   ├── 📁 core/                   # Core mathematical components
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 problem.py          # ✅ Problem definition class
│   │   │   ├── 📄 discretization.py  # Finite element discretization
│   │   │   ├── 📄 constraints.py     # Boundary/interface constraints
│   │   │   ├── 📄 static_condensation_ooc.py  # Static condensation
│   │   │   └── 📄 bulk_data.py        # Bulk solution management
│   │   │
│   │   ├── 📁 geometry/               # ✅ Geometry management module
│   │   │   ├── 📄 __init__.py         # ✅ Exports DomainGeometry, DomainInfo
│   │   │   └── 📄 domain_geometry.py  # ✅ Multi-domain geometry class
│   │   │
│   │   ├── 📁 problems/               # Problem definition library
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 ooc_test_problem.py        # Basic OoC 4-equation system
│   │   │   ├── 📄 KS_traveling_wave.py       # Keller-Segel analytical solution
│   │   │   ├── 📄 T_junction.py              # ✅ T-junction network
│   │   │   ├── 📄 KS_with_geometry.py        # ✅ KS with DomainGeometry
│   │   │   ├── 📄 KS_grid_geometry.py        # ✅ KS on grid network
│   │   │   └── 📄 OoC_grid_geometry.py       # ✅ OoC on grid network
│   │   │
│   │   ├── 📁 solver/                 # Numerical solver components
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 global_assembler.py        # Global system assembly
│   │   │   ├── 📄 newton_solver.py           # Newton-Raphson solver
│   │   │   └── 📄 time_integrator.py         # Time stepping methods
│   │   │
│   │   └── 📁 visualization/          # ✅ Plotting and visualization
│   │       ├── 📄 __init__.py
│   │       └── 📄 lean_matplotlib_plotter.py # ✅ Multi-mode plotter
│   │
│   ├── 📄 setup_solver.py             # Main solver setup interface
│   ├── 📄 test_evolution+plotting.py  # ✅ Main test/demo script
│   ├── 📄 test_geometry.py            # ✅ Geometry module tests
│   └── 📄 test_problem.py             # ✅ Problem module tests
│
├── 📁 examples/                       # Example applications
│   └── 📄 keller_segel_example.py     # ✅ Basic KS setup example
│
├── 📁 docs/                           # ✅ Documentation
│   ├── 📄 BioNetFlux_Documentation.md     # ✅ Main documentation (Markdown)
│   ├── 📄 BioNetFlux_Documentation.tex    # ✅ Main documentation (LaTeX)
│   ├── 📄 Mathematical_Background.md      # ✅ Mathematical theory (Markdown)
│   ├── 📄 Mathematical_Background.tex     # ✅ Mathematical theory (LaTeX)
│   ├── 📄 Code_Structure_Schematic.md     # ✅ Code structure diagram
│   ├── 📄 Current_Code_Structure.md       # ✅ This file
│   ├── 📄 compile_documentation.sh        # ✅ LaTeX compilation script
│   └── 📄 README_latex.md                 # ✅ LaTeX compilation guide
│
├── 📁 Logos/                          # Brand assets
│   ├── 🖼️ BioNetFlux.png              # Main logo
│   └── 🖼️ Barra.png                   # Institution bar
│
├── 📄 README.md                       # ✅ Project overview
└── 📄 .gitignore                      # ✅ Git ignore rules

✅ = Confirmed implemented/updated
```

## Current Implementation Status

### ✅ Fully Implemented Components

#### Core Framework (`ooc1d/core/`)

**`problem.py`** - Enhanced Problem class
- ✅ Basic problem definition with validation
- ✅ Self-testing capabilities (`validate_problem()`, `test_functions()`)
- ✅ Dynamic function setting (`set_function()`)
- ✅ Factory method for test problems (`create_test_problems()`)
- ✅ Comprehensive self-test suite (`run_self_test()`)
- ✅ Support for multiple problem types (KS, OoC, generic)

#### Geometry System (`ooc1d/geometry/`)

**`domain_geometry.py`** - Multi-domain network management
- ✅ `DomainGeometry` class for network definition
- ✅ `DomainInfo` dataclass for domain properties
- ✅ Proper segment-segment intersection detection (fixed)
- ✅ Connectivity analysis (`get_connectivity_info()`)
- ✅ Parameter space management (`suggest_parameter_spacing()`)
- ✅ Comprehensive validation (`validate_geometry()`)
- ✅ Factory method for test geometries (`create_test_geometries()`)
- ✅ Self-testing capabilities (`run_self_test()`)

#### Visualization System (`ooc1d/visualization/`)

**`lean_matplotlib_plotter.py`** - Multi-mode plotting
- ✅ 2D curve plots (separate subplot per domain)
- ✅ Flat 3D view with rounded segment ends
- ✅ Bird's eye view without domain labels
- ✅ Comparison plots (initial vs final)
- ✅ Automatic equation name detection
- ✅ Flexible save/display options

#### Problem Library (`ooc1d/problems/`)

**Current Problem Definitions:**
- ✅ `T_junction.py` - Two-domain T-junction with Kedem-Katchalsky constraints
- ✅ `KS_with_geometry.py` - Keller-Segel using DomainGeometry class
- ✅ `KS_grid_geometry.py` - KS on complex grid network (2 verticals + 4 horizontals)
- ✅ `OoC_grid_geometry.py` - Organ-on-chip on same grid with species-specific permeabilities

### ✅ Testing Framework

**Comprehensive Test Suite:**
- ✅ `test_problem.py` - Problem class validation (moved to code/ directory)
- ✅ `test_geometry.py` - Geometry module validation
- ✅ `test_evolution+plotting.py` - Full pipeline integration test
- ✅ Self-testing built into core classes
- ✅ Performance benchmarking
- ✅ Error handling validation

### ✅ Documentation System

**Complete Documentation:**
- ✅ Markdown documentation with API reference
- ✅ LaTeX documentation with mathematical background
- ✅ Mathematical theory for KS and OoC models
- ✅ Compilation scripts and guides
- ✅ Usage examples and tutorials

### ✅ Examples and Tutorials

**`examples/keller_segel_example.py`**
- ✅ Basic Keller-Segel problem setup
- ✅ Demonstrates core Problem class usage
- ✅ Shows function setting patterns

## Key Features and Capabilities

### 🏗️ Architecture Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Multi-domain support** | ✅ Complete | Complex network geometries |
| **Geometry management** | ✅ Complete | DomainGeometry class with validation |
| **Problem validation** | ✅ Complete | Built-in validation and testing |
| **Visualization modes** | ✅ Complete | 2D curves, 3D flat, bird's eye view |
| **Interface constraints** | ✅ Complete | Neumann, continuity, Kedem-Katchalsky |
| **Self-testing** | ✅ Complete | All modules include self-validation |

### 🧮 Mathematical Models

| Model Type | Implementation | Features |
|------------|---------------|----------|
| **Keller-Segel** | ✅ Complete | Chemotaxis, analytical solutions |
| **Organ-on-Chip** | ✅ Complete | 4-equation system, species transport |
| **Generic PDEs** | ✅ Complete | Flexible equation systems |

### 🗺️ Network Topologies

| Topology | Status | Description |
|----------|--------|-------------|
| **Linear chains** | ✅ Complete | Sequential domain connections |
| **T-junctions** | ✅ Complete | Three-way intersections |
| **Grid networks** | ✅ Complete | Complex rectangular grids |
| **Star networks** | ✅ Complete | Radial configurations |
| **Branching networks** | ✅ Complete | Tree-like structures |

### 📊 Visualization Capabilities

| Plot Type | Status | Use Case |
|-----------|--------|----------|
| **Domain profiles** | ✅ Complete | Solution vs position per domain |
| **Network 3D view** | ✅ Complete | Topology with solution heights |
| **Bird's eye view** | ✅ Complete | Network overview with color coding |
| **Time evolution** | ✅ Complete | Initial vs final comparisons |

## Current Usage Workflow

### 1. Problem Definition
```python
from bionetflux.problems import KS_grid_geometry
problems, global_disc, constraints, name = KS_grid_geometry.create_global_framework()
```

### 2. Solver Setup
```python
from setup_solver import quick_setup
setup = quick_setup("ooc1d.problems.KS_grid_geometry", validate=True)
```

### 3. Initial Conditions
```python
trace_solutions, multipliers = setup.create_initial_conditions()
```

### 4. Visualization
```python
from bionetflux.visualization import LeanMatplotlibPlotter
plotter = LeanMatplotlibPlotter(problems, discretizations)
plotter.plot_2d_curves(trace_solutions)
plotter.plot_birdview(trace_solutions, equation_idx=0, time=0.0)
```

### 5. Time Evolution
```python
# Newton iteration loop with global assembler
# (Full implementation in test_evolution+plotting.py)
```

## File Dependencies

```
📄 test_evolution+plotting.py
├── setup_solver.py
├── ooc1d.visualization.lean_matplotlib_plotter
└── ooc1d.problems.* (configurable)

📄 setup_solver.py
├── ooc1d.core.problem
├── ooc1d.core.discretization
├── ooc1d.core.constraints
├── ooc1d.solver.global_assembler
└── ooc1d.core.bulk_data

📁 ooc1d.problems.*
├── ooc1d.core.problem
├── ooc1d.geometry.domain_geometry
├── ooc1d.core.discretization
└── ooc1d.core.constraints

📄 test_geometry.py
└── ooc1d.geometry.domain_geometry

📄 test_problem.py
└── ooc1d.core.problem
```

## Recent Enhancements

### Geometry Module Improvements
- ✅ Fixed segment intersection detection (was only checking endpoints)
- ✅ Added comprehensive validation with warnings vs errors
- ✅ Improved connectivity analysis with component detection
- ✅ Added factory methods for standard test geometries

### Problem Class Enhancements
- ✅ Added comprehensive validation and function testing
- ✅ Implemented self-testing capabilities
- ✅ Added support for dynamic function setting
- ✅ Enhanced error handling and edge case management

### Visualization Improvements
- ✅ Separated domains into individual subplots for 2D curves
- ✅ Added rounded ends to 3D flat view segments
- ✅ Removed domain labels from bird's eye view
- ✅ Updated title format to include time information

### Testing Framework
- ✅ Comprehensive test suites for all major components
- ✅ Performance benchmarking capabilities
- ✅ Built-in self-validation for core classes
- ✅ Error handling and edge case testing

## Integration Points

### Problem → Geometry
- Problems use `DomainGeometry` to define network topology
- Extrema coordinates set from geometry for visualization
- Parameter spaces managed through geometry validation

### Solver → Visualization  
- `LeanMatplotlibPlotter` reads domain information from problems
- Trace solutions passed directly from solver to plotter
- Multiple visualization modes for different analysis needs

### Testing → All Modules
- Each major component includes self-testing capabilities
- Dedicated test scripts for integration testing
- Performance monitoring and validation

This structure represents a mature, well-tested framework with comprehensive documentation, multiple problem types, flexible geometry management, and powerful visualization capabilities.
