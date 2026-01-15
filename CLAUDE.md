# Skills Library Design Guide

This document describes the architecture and conventions for the role-based skills library.

## Overview

The skills library provides role-specific skill catalogs that Claude can reference to assist with domain-specific tasks. Each role has ~100 targeted skills organized by category, plus 25 shared cross-functional skills.

**How it works:**
1. User mentions a role or task domain
2. Claude references the relevant catalog
3. User can request details on specific skills (e.g., "Give me details on CV42")
4. Claude generates implementation guidance on-demand

## Directory Structure

```
skills/
├── CLAUDE.md              # This file (design guide)
├── README.md              # Public overview
├── ROLE_TEMPLATE.md       # Template for creating new roles
├── _shared/
│   ├── catalog.md         # 25 shared skills (S01-S25)
│   └── TEMPLATE.md        # Skill implementation template
├── ml-engineer/
│   └── catalog.md         # 100 ML skills (M01-M100)
├── cv-engineer/
│   └── catalog.md         # 100 CV skills (CV01-CV100)
└── <new-role>/
    └── catalog.md         # 100 role skills (<PREFIX>01-<PREFIX>100)
```

## Design Principles

| Principle | Description |
|-----------|-------------|
| **100 + 25** | Each role has ~100 specific skills + 25 shared skills |
| **Catalog-only** | Store skill lists, not implementations (generate on-demand) |
| **ID prefixes** | 2-letter prefix per role (M, CV, BE, etc.) |
| **Categories** | 5-12 logical groupings per role |
| **On-demand details** | Users request implementation via "Details on X" |

## Existing Roles

| Role | Prefix | Skills | Focus Areas |
|------|--------|--------|-------------|
| ML Engineer | `M` | M01-M100 | Model training, MLOps, deployment, experimentation |
| CV Engineer | `CV` | CV01-CV100 | Image/video processing, classical CV, restoration |
| Shared | `S` | S01-S25 | Git, docs, testing, DevOps, code quality |

## Adding a New Role

### Step 1: Create Directory
```bash
mkdir skills/<role-name>
```

### Step 2: Choose ID Prefix
Pick a unique 2-letter prefix:
- Not already used (M, CV, S are taken)
- Intuitive abbreviation of role name
- Examples: BE (Backend), FE (Frontend), DE (Data), DO (DevOps)

### Step 3: Create Catalog
Copy structure from `ROLE_TEMPLATE.md`:

```markdown
# <Role Name> Skills Catalog

100 skills for <Role Name>. Request details on specific skills as needed.

**Also uses:** 25 shared skills from `_shared/catalog.md`

---

## 1. Category Name (X skills)

| ID | Name | Description |
|----|------|-------------|
| XX01 | skill-name | Brief one-line description |

---

## Summary

| Category | Count | IDs |
|----------|-------|-----|
| Category 1 | 15 | XX01-XX15 |
| **Total** | **100** | |

---

## Request Details

To get detailed implementation for any skill, ask:
> "Give me details on XX01 (skill-name)"
```

### Step 4: Organize Categories
- Target 5-12 categories
- Distribute skills evenly (~8-15 per category)
- Group by workflow or domain area
- Order from foundational to advanced

### Step 5: Write Skills
For each skill:
- **ID**: `<PREFIX><##>` (e.g., BE01, BE02)
- **Name**: kebab-case verb phrase (e.g., `connection-pooler`)
- **Description**: One line, action-oriented

### Step 6: Add Summary Table
Include at end of catalog:
```markdown
| Category | Count | IDs |
|----------|-------|-----|
| API Design | 12 | BE01-BE12 |
| Database | 15 | BE13-BE27 |
| ...
| **Total** | **100** | |
```

## Catalog Format Conventions

### ID Naming
```
<PREFIX><##>
   │     │
   │     └── Two-digit number (01-99)
   └── Two-letter role prefix (unique)
```

Examples: `M31`, `CV42`, `S15`, `BE07`

### Table Format
```markdown
| ID | Name | Description |
|----|------|-------------|
| XX01 | kebab-case-name | Brief description (one line) |
```

### Skill Naming
- Use kebab-case: `api-rate-limiter`, `database-optimizer`
- Start with action/noun: `cache-manager`, `query-builder`
- Be specific: `jwt-authenticator` not `auth-handler`

### Description Guidelines
- One line only
- Action-oriented ("Configure X", "Implement Y", "Debug Z")
- Include key technologies in parentheses when helpful

## Shared Skills Reference

All roles inherit 25 shared skills from `_shared/catalog.md`:

| Category | IDs | Examples |
|----------|-----|----------|
| Git & Version Control | S01-S05 | commit-generator, pr-description-writer |
| Documentation | S06-S10 | readme-generator, docstring-writer |
| Testing | S11-S15 | unit-test-generator, mock-data-generator |
| DevOps & Deployment | S16-S20 | dockerfile-generator, github-actions-creator |
| Code Quality | S21-S25 | code-reviewer, security-scanner |

## Future Role Ideas

| Role | Suggested Prefix | Potential Categories |
|------|------------------|---------------------|
| Backend Engineer | BE | API, Database, Auth, Caching, Messaging |
| Frontend Engineer | FE | Components, State, Styling, Performance, A11y |
| DevOps Engineer | DO | CI/CD, Containers, Cloud, Monitoring, IaC |
| Data Engineer | DE | Pipelines, Warehousing, Streaming, Quality |
| Security Engineer | SE | AppSec, NetSec, Crypto, Compliance, Incident |
| Mobile Engineer | MO | iOS, Android, Cross-platform, Push, Offline |

## On-Demand Skill Details

When user requests details, use `_shared/TEMPLATE.md` format:

```yaml
---
name: skill-name
description: Brief description
category: category-name
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Skill Name

## When to Use
- Condition 1
- Condition 2

## Instructions
1. Step 1
2. Step 2

## Example
```code```

## Best Practices
- Practice 1
- Practice 2

## Related Skills
- related-skill-1
- related-skill-2
```

## Versioning & Changes

- **Never renumber** existing skill IDs
- Append new skills with next available ID
- Use CHANGELOG.md for tracking changes
- Document decisions in CHANGELOG_DETAILED.md
