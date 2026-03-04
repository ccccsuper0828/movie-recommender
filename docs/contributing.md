# Contributing to CineMatch

Thank you for your interest in contributing to CineMatch! This document provides guidelines for contributing to the project.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Team Roles](#team-roles)

## Getting Started

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

## Development Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Set Up Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### 4. Download Data
```bash
# Place TMDB dataset files in data/raw/
# - tmdb_5000_movies.csv
# - tmdb_5000_credits.csv
```

### 5. Verify Setup
```bash
# Run tests
pytest tests/ -v

# Start the frontend
streamlit run frontend/app.py

# Start the API
uvicorn api.main:app --reload
```

## Project Structure

```
MOVIE/
├── config/           # Configuration (Settings lead)
├── src/
│   ├── core/         # Algorithms (Algorithm Developer)
│   ├── data/         # Data handling (Data/Infra Developer)
│   ├── explainability/  # Explanations (Explainability Developer)
│   ├── models/       # Data models
│   ├── services/     # Business logic
│   └── utils/        # Utilities
├── api/              # REST API (Backend Developer)
├── frontend/         # UI (Frontend Developer)
├── analytics/        # Analytics (Explainability Developer)
├── tests/            # Tests (Testing Lead)
└── docs/             # Documentation (Testing Lead)
```

## Code Style

### Python Style Guide
We follow PEP 8 with the following additions:

```python
# Use type hints
def recommend(self, title: str, n: int = 10) -> pd.DataFrame:
    """
    Get movie recommendations.

    Parameters
    ----------
    title : str
        Source movie title
    n : int
        Number of recommendations

    Returns
    -------
    pd.DataFrame
        Recommendations with similarity scores
    """
    pass

# Use docstrings in NumPy style
# Use 4 spaces for indentation
# Maximum line length: 88 characters (Black formatter)
```

### Formatting Tools
```bash
# Format code
black src/ api/ frontend/ tests/

# Sort imports
isort src/ api/ frontend/ tests/

# Lint code
flake8 src/ api/ frontend/ tests/

# Type checking
mypy src/ api/
```

### Naming Conventions
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

## Making Changes

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### Branch Naming
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `test/` - Test additions

### 2. Make Your Changes

Follow these guidelines:
- Write clean, readable code
- Add docstrings to all functions/classes
- Update tests for your changes
- Update documentation if needed

### 3. Commit Your Changes
```bash
git add .
git commit -m "feat: add new recommendation method"
```

#### Commit Message Format
```
type: subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

## Testing

### Running Tests
```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_content_based.py -v
```

### Writing Tests
```python
# tests/unit/test_my_feature.py

import pytest
from src.core import MyFeature


class TestMyFeature:
    """Tests for MyFeature."""

    def test_basic_functionality(self, sample_data):
        """Test that basic functionality works."""
        feature = MyFeature()
        result = feature.process(sample_data)

        assert result is not None
        assert len(result) > 0

    def test_edge_case(self):
        """Test edge case handling."""
        feature = MyFeature()

        with pytest.raises(ValueError):
            feature.process(None)
```

### Test Coverage Requirements
- New code should have >= 80% test coverage
- Critical paths require >= 90% coverage

## Pull Request Process

### 1. Update Your Branch
```bash
git fetch origin
git rebase origin/main
```

### 2. Push Changes
```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

Include in your PR description:
- What changes were made
- Why the changes were made
- How to test the changes
- Screenshots (for UI changes)

### 4. PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### 5. Code Review

PRs require:
- At least 1 approval
- All CI checks passing
- No merge conflicts

## Team Roles

### Backend/API Developer
**Responsibilities:**
- FastAPI endpoints
- Authentication/authorization
- API middleware

**Key Files:**
- `api/main.py`
- `api/routes/*.py`
- `api/dependencies.py`

### Algorithm Developer
**Responsibilities:**
- Recommendation algorithms
- Similarity computations
- Performance optimization

**Key Files:**
- `src/core/*.py`

### Explainability Developer
**Responsibilities:**
- SHAP integration
- Rule-based explanations
- Analytics dashboard

**Key Files:**
- `src/explainability/*.py`
- `analytics/*.py`

### Frontend Developer
**Responsibilities:**
- Streamlit UI
- Component design
- CSS theming

**Key Files:**
- `frontend/app.py`
- `frontend/components/*.py`
- `frontend/styles/*.py`

### Data/Infra Developer
**Responsibilities:**
- Data pipeline
- Configuration system
- Docker/deployment

**Key Files:**
- `src/data/*.py`
- `config/*.py`
- `Dockerfile`

### Testing/Docs Lead
**Responsibilities:**
- Test suite maintenance
- CI/CD pipeline
- Documentation

**Key Files:**
- `tests/*.py`
- `docs/*.md`
- `.github/workflows/*.yml`

## Communication

### Channels
- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, ideas
- Pull Requests: Code reviews

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## Release Process

### Version Numbering
We use [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH
- Example: 1.2.3

### Release Checklist
1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Deploy to production

## Questions?

If you have questions:
1. Check existing documentation
2. Search GitHub issues
3. Open a new issue with the `question` label

Thank you for contributing!
