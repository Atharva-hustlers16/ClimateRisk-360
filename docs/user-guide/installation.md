# Installation

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ClimateRisk360.git
   cd ClimateRisk360
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

4. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

## Verifying the Installation

Run the test suite to ensure everything is working correctly:

```bash
pytest tests/ -v
```

You should see all tests passing. If you encounter any issues, please check the [troubleshooting](troubleshooting.md) guide.
