# Skills Changelog (Detailed)

## [Unreleased]

### Added
- Best Practices section to ml-inference skill (weight loading, path handling, model init, dependency isolation, multi-model, testing) #skill #ml-inference (2026-01-14) [#63ab]
  - Files: `~/.claude/skills/ml-inference.md`
- `~/.claude/RU_CHANGELOG.md` for global RU conversion tracking #skill #ml-inference (2026-01-14) [#63ab]
  - Files: `~/.claude/RU_CHANGELOG.md`

### Changed
- Moved ml-inference skill from project-level to user-level for global access #skill #ml-inference (2026-01-14) [#63ab]
  - Files: `~/.claude/skills/ml-inference.md` (was `.claude/skills/ml-inference.md`)
- Updated workflow from 4 to 5 steps (added Document step) #skill #ml-inference (2026-01-14) [#63ab]
- Consolidated Common Fixes into table format #skill #ml-inference (2026-01-14) [#63ab]
- Streamlined v1.1.0 reproducibility section #skill #ml-inference (2026-01-14) [#63ab]

### Discussed
- User-level vs project-level skill storage strategy #planning (2026-01-14) [#63ab]

### Learned
- **Decision**: User-level skills (`~/.claude/skills/`) for global access vs project-level for project-specific - chose user-level for reusable skills like ml-inference #skill #config (2026-01-14) [#63ab]
- **Pattern**: RU_CHANGELOG tracks conversions with challenges/learnings sections for knowledge preservation across sessions #ml-inference #documentation (2026-01-14) [#63ab]
- **Gotcha**: Sandbox restricts writes to `~/.claude/` in some cases - may need manual file operations for cleanup #sandbox #config (2026-01-14) [#63ab]
