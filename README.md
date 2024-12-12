# Multi-Agent Framework

A hierarchical, multi-agent neuro-symbolic AI framework for software development. The framework incorporates Prolog for logical reasoning, formal methods for verification, and advanced prompting techniques.

## Features

- Pointcut-based aspect-oriented programming
- Web-based management dashboard
- Pattern validation and suggestions
- Caching and performance optimization
- Comprehensive test coverage

## Project Structure

```
multi_agent_framework/
├── core/
│   └── pointcuts/
│       └── manager.py      # Core pointcut functionality
├── web/
│   ├── app.py             # FastAPI application
│   ├── routes/
│   │   └── pointcut_routes.py
│   ├── static/
│   └── templates/
│       └── pointcut_dashboard.html
└── tests/
    ├── test_pointcut_manager.py
    └── test_pointcut_routes.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-agent-framework.git
cd multi-agent-framework
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

Run the test suite:
```bash
pytest tests/
```

## Starting the Dashboard

1. Start the FastAPI server:
```bash
uvicorn multi_agent_framework.web.app:app --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

### Creating Pointcuts

```python
from multi_agent_framework.core.pointcuts.manager import PointcutManager, PointcutType

# Initialize manager
manager = PointcutManager()

# Add a pointcut
pointcut_id = manager.add_pointcut(
    pattern="test_.*",
    pointcut_type=PointcutType.METHOD_EXECUTION,
    metadata={"description": "Test methods"}
)

# Check matches
matches = manager.check_matches("test_function")
```

### Using the Web Dashboard

1. Navigate to the dashboard
2. Use the pattern editor to create new pointcuts
3. View and manage existing pointcuts
4. Enable/disable pointcuts as needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
