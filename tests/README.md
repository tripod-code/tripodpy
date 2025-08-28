# TriPoD Unit Testing

This directory contains unit tests for the TriPoD (Tri-population method for dust evolution in protoplanetary disks) package.

## Setup

### Prerequisites

```bash
pip install pytest pytest-cov numpy scipy
```

### Running Tests

#### Option 1: Using pytest directly
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_size_distribution_standalone.py -v

# Run tests with coverage
pytest tests/unit/ --cov=tripod --cov-report=term-missing
```

#### Option 2: Using the test runner
```bash
cd tests
python test_runner.py
```

## Test Structure

### Unit Tests

The unit tests focus on testing individual functions in isolation:

#### `test_size_distribution_standalone.py`
Tests for functions in `tripod/utils/size_distribution.py`:

- **`get_rhos_simple()`**: Tests bulk density computation for particle size distributions
- **`get_q()`**: Tests power law exponent calculation from surface densities
- **`get_size_distribution()`**: Tests power-law size distribution generation with various parameters
- **`average_size()`**: Tests average size calculations for power-law distributions

**Key test categories:**
- Basic functionality tests
- Edge cases (empty arrays, equal values, special q values)
- Parameter validation
- Mathematical consistency checks
- Integration tests combining multiple functions

#### `test_read_data.py` 
Tests for functions in `tripod/utils/read_data.py`:

- **`read_data()`**: Tests data reading and processing from simulation objects and files
- Mock-based testing to avoid dependency on dustpy
- Parameter validation and error handling

## Functions Suitable for Unit Testing

### High Priority (Pure Mathematical Functions)

1. **`tripod/utils/size_distribution.py`**:
   - ✅ `get_rhos_simple()` - Computes bulk density (pure function)
   - ✅ `get_q()` - Calculates power law exponent (pure function) 
   - ✅ `get_size_distribution()` - Generates size distributions (well-defined I/O)
   - ✅ `average_size()` - Computes average size of distributions (pure function)

2. **`tripod/utils/read_data.py`**:
   - ✅ `read_data()` - Data processing and reconstruction (can be mocked)

### Medium Priority (Requires Simulation Context)

3. **`tripod/std/sim.py`**:
   - `dt()` - Time step calculation (needs simulation state)
   - `prepare_implicit_dust()` - Dust preparation (simulation-dependent)
   - `finalize_implicit_dust()` - Dust finalization (simulation-dependent)

4. **`tripod/simulation.py`**:
   - Individual initialization methods (require complex setup)
   - Grid generation methods (mathematical but simulation-dependent)

### Lower Priority (Integration/System Level)

5. **Full simulation runs**: Better suited for integration tests
6. **I/O operations**: Better suited for integration tests with real files

## Test Design Principles

1. **Isolation**: Unit tests test individual functions without dependencies
2. **Deterministic**: Tests use fixed inputs and expected outputs
3. **Fast**: Tests run quickly without heavy computation or I/O
4. **Comprehensive**: Tests cover normal cases, edge cases, and error conditions
5. **Mathematical Validation**: Tests verify mathematical properties (conservation, bounds, etc.)

## Running Integration Tests

The repository also contains integration tests in the main `tests/` directory:
- `DuffellGap.py` - Gap formation scenario
- `Smooth.py` - Smooth disk evolution
- `Icelines.py` - Ice line physics
- `DP_DuffellGap.py` - DustPy comparison

These require the full dustpy installation and are better suited for CI/system validation.

## Extending Tests

When adding new testable functions:

1. **Pure functions** (no side effects): Add to existing test files or create new ones
2. **Functions with dependencies**: Use mocking to isolate the function under test
3. **Mathematical functions**: Include property-based tests (conservation, bounds, monotonicity)
4. **Complex functions**: Break down into smaller testable components

### Example Test Structure
```python
class TestNewFunction:
    def test_basic_functionality(self):
        """Test normal operation."""
        pass
        
    def test_edge_cases(self):
        """Test boundary conditions.""" 
        pass
        
    def test_error_conditions(self):
        """Test invalid inputs."""
        pass
        
    def test_mathematical_properties(self):
        """Test conservation, bounds, etc."""
        pass
```

## Coverage Goals

- **High coverage** (>90%) for utility functions
- **Medium coverage** (>70%) for simulation initialization functions  
- **Basic coverage** (>50%) for complex simulation dynamics

Current coverage focuses on the most testable and critical utility functions that form the mathematical foundation of the simulation.