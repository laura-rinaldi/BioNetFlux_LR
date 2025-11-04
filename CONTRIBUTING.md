# Contributing to BioNetFlux

Thank you for your interest in contributing to BioNetFlux! This guide will help you set up your development environment and understand our testing practices.

## Development Environment Setup

### 1. Fork and Clone
```bash
git fork https://github.com/silvia-bertoluzza/bionetflux
git clone https://github.com/YOUR_USERNAME/bionetflux.git
cd bionetflux
```

### 2. Create Development Environment

#### Option A: Using Conda (Recommended)
```bash
conda create -n bionetflux-dev python=3.11
conda activate bionetflux-dev
pip install -r requirements-dev.txt
```

#### Option B: Using venv
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 3. Configure VS Code

1. **Copy template settings**:
   ```bash
   cp .vscode/settings.json.template .vscode/settings.json
   ```

2. **Set Python interpreter** in VS Code:
   - `Cmd+Shift+P` → "Python: Select Interpreter"
   - Choose your virtual environment Python

3. **Install recommended extensions**:
   - Python (ms-python.python)
   - Pylance (ms-python.vscode-pylance) 
   - Python Test Explorer (littlefoxteam.vscode-python-test-adapter)

## Testing Environment

### Test Structure
```
tests/
├── __init__.py
├── test_sample.py          # Basic functionality tests
├── test_geometry.py        # Geometry module tests
├── test_problem.py         # Problem definition tests
├── test_bulk_data.py       # Data management tests
└── ...                     # Additional test modules
```

### Running Tests

#### Command Line
```bash
# All tests
pytest

# Specific test file
pytest tests/test_geometry.py

# Specific test function
pytest tests/test_geometry.py::TestBasicFunctionality::test_empty_geometry_creation

# With coverage report
pytest --cov=src/bionetflux --cov-report=html

# Verbose output
pytest -v
```

#### VS Code Integration
- Tests appear in Explorer sidebar under "Test Explorer"
- Click play button (▶️) next to any test to run
- Use debug icon (🐛) for debugging tests
- Right-click for additional options

### Test Configuration

The project uses pytest with these configurations:

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

**VS Code settings.json**:
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests", "-v", "--tb=short"],
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "python.analysis.extraPaths": ["./src"]
}
```

## Writing Tests

### Test Guidelines

1. **File Naming**: `test_*.py` for test files
2. **Function Naming**: `test_*` for test functions  
3. **Class Naming**: `Test*` for test classes
4. **Imports**: Always use relative imports from `src/`

### Example Test Structure
```python
"""Test module for geometry functionality."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bionetflux.geometry.domain_geometry import DomainGeometry

class TestGeometryBasics:
    """Test basic geometry functionality."""
    
    def test_empty_geometry_creation(self):
        """Test creating empty geometry."""
        geometry = DomainGeometry("test_network")
        assert geometry.name == "test_network"
        assert len(geometry.domains) == 0
    
    @pytest.mark.slow
    def test_large_geometry_performance(self):
        """Test performance with large geometries."""
        # Implementation for slow tests
        pass
```

### Test Markers
- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Tests involving multiple components  
- `@pytest.mark.slow`: Long-running tests

### Fixtures
Use pytest fixtures for common test setup:
```python
@pytest.fixture
def simple_geometry():
    """Create a simple test geometry."""
    geometry = DomainGeometry("test")
    geometry.add_domain((0,0), (1,0), "segment1")
    return geometry
```

## Code Quality

### Style Guidelines
- Follow PEP 8
- Use type hints where possible
- Document functions with docstrings
- Keep functions focused and small

### Running Quality Checks
```bash
# Linting
flake8 src/ tests/

# Type checking (if using mypy)
mypy src/

# Code formatting (if using black)
black src/ tests/
```

## Submitting Changes

### Pull Request Process
1. Create feature branch: `git checkout -b feature/new-feature`
2. Write tests for new functionality
3. Ensure all tests pass: `pytest`
4. Update documentation if needed
5. Submit pull request with clear description

### Commit Messages
```
feat: add new geometry validation function

- Implement domain overlap detection
- Add comprehensive test suite
- Update documentation with examples
```

## Troubleshooting

### Common Issues

**Tests not discovered in VS Code**:
1. Check Python interpreter is set correctly
2. Verify `.vscode/settings.json` configuration
3. Reload VS Code window: `Cmd+Shift+P` → "Developer: Reload Window"

**Import errors**:
1. Ensure `src/` is in Python path
2. Check virtual environment activation
3. Verify relative imports in test files

**Environment inconsistencies**:
1. Use `requirements-dev.txt` for exact versions
2. Document any platform-specific requirements
3. Consider using conda environment files

## Questions?

- Check existing [Issues](https://github.com/silvia-bertoluzza/bionetflux/issues)
- Create new issue for bugs or feature requests
- Review [Documentation](docs/) for API details

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.