# Skills Changelog

## 2026-01-14

### Session: Enhanced ml-inference skill with best practices and moved to user-level [#63ab]
- **Changed**: Moved `ml-inference` skill from project-level to user-level (`~/.claude/skills/ml-inference.md`) for global access #skill #ml-inference
- **Added**: Best Practices section with 6 categories: weight loading, path handling, model initialization, dependency isolation, multi-model support, testing #skill #ml-inference
- **Added**: `~/.claude/RU_CHANGELOG.md` for global RU conversion tracking with 5 backfilled entries #skill #ml-inference
- **Changed**: Updated workflow from 4 to 5 steps (added "Document" step for changelog updates) #skill #ml-inference
- **Changed**: Consolidated Common Fixes into table format for readability #skill #ml-inference
- **Changed**: Streamlined v1.1.0 reproducibility section #skill #ml-inference
- **Discussed**: User-level vs project-level skill storage - decided user-level for global access, project-level RU_CHANGELOG for project-specific tracking (later consolidated to user-level) #planning

---
