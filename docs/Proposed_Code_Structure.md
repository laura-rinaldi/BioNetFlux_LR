# BioNetFlux Proposed Code Structure (Version 1)

*A conservative reorganization focusing only on existing modules with minimal changes*

## Current vs Proposed Structure

### Current Structure
```
BioNetFlux/
├── 📁 code/
│   ├── 📁 ooc1d/
│   │   ├── 📁 core/
│   │   ├── 📁 geometry/
│   │   ├── 📁 problems/
│   │   ├── 📁 solver/
│   │   └── 📁 visualization/
│   ├── 📄 setup_solver.py
│   ├── 📄 test_evolution+plotting.py
│   ├── 📄 test_geometry.py
│   └── 📄 test_problem.py
├── 📁 examples/
└── 📁 docs/
```

### Proposed Structure (Version 1)
```
BioNetFlux/
├── 📁 src/                            # Renamed from 'code/'
│   ├── 📁 bionetflux/                 # Renamed from 'ooc1d/'
│   │   ├── 📄 __init__.py             # Package initialization
│   │   │
│   │   ├── 📁 core/                   # Existing core modules
│   │   │   ├── 📄 __init__.py         
│   │   │   ├── 📄 problem.py          # ✅ Existing
│   │   │   ├── 📄 discretization.py  # ✅ Existing
│   │   │   ├── 📄 constraints.py     # ✅ Existing
│   │   │   ├── 📄 static_condensation.py  # Renamed from static_condensation_ooc.py
│   │   │   └── 📄 bulk_data.py        # ✅ Existing
│   │   │
│   │   ├── 📁 geometry/               # Existing geometry module
│   │   │   ├── 📄 __init__.py         # ✅ Existing
│   │   │   └── 📄 domain_geometry.py  # ✅ Existing
│   │   │
│   │   ├── 📁 problems/               # Existing problem definitions
│   │   │   ├── 📄 __init__.py         
│   │   │   ├── 📄 ooc_test_problem.py        # ✅ Existing
│   │   │   ├── 📄 KS_traveling_wave.py       # ✅ Existing
│   │   │   ├── 📄 T_junction.py              # ✅ Existing
│   │   │   ├── 📄 KS_with_geometry.py        # ✅ Existing
│   │   │   ├── 📄 KS_grid_geometry.py        # ✅ Existing
│   │   │   └── 📄 OoC_grid_geometry.py       # ✅ Existing
│   │   │
│   │   ├── 📁 solver/                 # Existing solver module (keep name)
│   │   │   ├── 📄 __init__.py         
│   │   │   ├── 📄 global_assembler.py        # ✅ Existing
│   │   │   ├── 📄 newton_solver.py           # ✅ Existing
│   │   │   └── 📄 time_integrator.py         # ✅ Existing
│   │   │
│   │   └── 📁 visualization/          # Existing visualization
│   │       ├── 📄 __init__.py         
│   │       └── 📄 lean_matplotlib_plotter.py # ✅ Existing (keep name)
│   │
│   └── 📄 setup_solver.py             # ✅ Existing (moved to src/)
│
├── 📁 tests/                          # Reorganized tests
│   ├── 📄 test_problem.py             # Moved from code/
│   ├── 📄 test_geometry.py            # Moved from code/
│   └── 📄 test_evolution_plotting.py  # Renamed from test_evolution+plotting.py
│
├── 📁 examples/                       # Existing examples
│   └── 📄 keller_segel_example.py     # ✅ Existing
│
├── 📁 outputs/                        # New: organized outputs (git-ignored)
│   ├── 📁 plots/                      # Generated plots
│   └── 📁 data/                       # Simulation data
│
├── 📁 docs/                           # Existing documentation
│   ├── 📄 BioNetFlux_Documentation.md     # ✅ Existing
│   ├── 📄 BioNetFlux_Documentation.tex    # ✅ Existing
│   ├── 📄 Mathematical_Background.md      # ✅ Existing
│   ├── 📄 Mathematical_Background.tex     # ✅ Existing
│   └── ...                               # Other existing docs
│
├── 📁 Logos/                          # ✅ Existing
│   ├── 🖼️ BioNetFlux.png              
│   └── 🖼️ Barra.png                   
│
├── 📄 README.md                       # ✅ Existing
└── 📄 .gitignore                      # ✅ Existing (updated)
```

## Key Changes (Version 1 - Conservative)

### ✅ Simple Reorganization Only

| Change Type | Current | Proposed | Rationale |
|-------------|---------|----------|-----------|
| **Package rename** | `ooc1d` | `bionetflux` | More descriptive, matches project name |
| **Source directory** | `code/` | `src/` | Standard Python convention |
| **Test separation** | Mixed with source | Dedicated `tests/` | Cleaner separation |
| **Output organization** | Scattered | `outputs/` folder | Clean working directory |
| **File renames** | Minimal | Only 2 files | Reduce breaking changes |

### 🚫 What's NOT Changed (Postponed to v2)

- ❌ No hierarchical model organization
- ❌ No new utility modules
- ❌ No advanced test categorization
- ❌ No new configuration system
- ❌ No package restructuring beyond renaming
- ❌ No file content modifications

## Only Two File Renames

1. `static_condensation_ooc.py` → `static_condensation.py` (remove redundant suffix)
2. `test_evolution+plotting.py` → `test_evolution_plotting.py` (remove special character)

## Import Changes Required

### Before (Current)
```python
from bionetflux.core.problem import Problem
from bionetflux.geometry.domain_geometry import DomainGeometry
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from bionetflux.problems.KS_grid_geometry import create_global_framework
import setup_solver
```

### After (Proposed)
```python
from bionetflux.core.problem import Problem
from bionetflux.geometry.domain_geometry import DomainGeometry
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from bionetflux.problems.KS_grid_geometry import create_global_framework
import setup_solver  # Same - now in src/
```

## New Package Initialization

### `src/bionetflux/__init__.py`
```python
"""BioNetFlux: Multi-Domain Biological Network Flow Simulation Framework"""

__version__ = "1.0.0"

# Main exports for convenience
from .core.problem import Problem
from .geometry.domain_geometry import DomainGeometry, DomainInfo
from .visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

__all__ = ["Problem", "DomainGeometry", "DomainInfo", "LeanMatplotlibPlotter"]
```

## Benefits of Conservative Approach

### ✅ Advantages
- **Minimal risk**: Only import changes, no logic modifications
- **Easy rollback**: Simple to undo if issues arise
- **Gradual transition**: Users can adapt slowly
- **Preserve functionality**: All existing code continues to work
- **Git history**: File history preserved with `git mv`

### 🎯 Immediate Improvements
- **Professional naming**: `bionetflux` instead of `ooc1d`
- **Standard layout**: `src/` follows Python best practices
- **Cleaner workspace**: Outputs organized and git-ignored
- **Better testing**: Tests separated from source code
- **Easier imports**: More intuitive package names

## Migration Complexity: LOW

### Changes Required
1. **Directory moves**: 5 `git mv` operations
2. **File renames**: 2 files only
3. **Import updates**: Automated find-and-replace
4. **Path updates**: Test file paths only
5. **Documentation**: Update examples in docs

### No Changes Required
- ❌ No algorithm modifications
- ❌ No class restructuring  
- ❌ No API changes
- ❌ No new dependencies
- ❌ No configuration changes

## Version 2 Future Enhancements

The conservative v1 structure provides a solid foundation for future enhancements in v2:

### Planned for Version 2
- Hierarchical model organization (`models/keller_segel/`, `models/organ_on_chip/`)
- Advanced test categorization (`tests/unit/`, `tests/integration/`)
- Utility modules (`utils/`, `config/`)
- Enhanced documentation structure
- Modern packaging with `pyproject.toml`
- CI/CD integration

This approach allows us to:
1. **Validate the basic reorganization** with minimal risk
2. **Get user feedback** on the new structure
3. **Plan v2 enhancements** based on real usage
4. **Maintain stability** during the transition

The conservative v1 structure maintains all existing functionality while providing a cleaner, more professional organization that follows Python best practices.
