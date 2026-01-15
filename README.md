# Claude Code Skills Library

A modular skills library organized by role, with shared skills that can be used across roles.

## Structure

```
skills/
├── README.md              # This file
├── _shared/               # Common skills used across ALL roles
│   └── catalog.md         # Shared skills catalog
├── ml-engineer/           # Machine Learning Engineer
│   └── catalog.md         # 100 ML-specific skills
├── data-scientist/        # Data Scientist (future)
├── backend-engineer/      # Backend Engineer (future)
├── devops-engineer/       # DevOps Engineer (future)
└── frontend-engineer/     # Frontend Engineer (future)
```

## How to Use

1. **Browse catalogs** - Each role has a `catalog.md` listing all skills
2. **Request details** - Ask for detailed implementation of specific skills
3. **Mix & match** - Combine shared skills with role-specific skills

## Adding a New Role

1. Create folder: `skills/<role-name>/`
2. Create catalog: `skills/<role-name>/catalog.md`
3. List skills with: ID, Name, Description, Category, Shared (Y/N)
4. Reference shared skills from `_shared/catalog.md`

## Skill Format

Each skill in a catalog:
```
| ID | Name | Description | Category | Shared |
```

Detailed skills (when requested) follow the template in `_shared/TEMPLATE.md`.
