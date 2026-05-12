# PROJECT

## Git Setup
- CMD: git init
- CMD: git branch -M main
- CMD: git add .
- CND: git commit -m "initial commit"
- CND: git renite add origin <REPO_URL>
- CMD: git push -u origin main
- **CREATE A .gitignore FIRST**


## MAS Patterns Applied

### Coordination Patterns
- Orchestrator-Worker — orchestrator drives all decisions, workers execute on demand
- Sequential Pipeline — proposal enriched in fixed order: course → room → lecturer → policy
- Planner-Generator-Evaluator — orchestrator plans the fix, worker regenerates, 
  policy evaluates, loops with failure_context until approved or abandoned
- Role-Based Agent Design — each agent owns exactly one domain, no overlap

### Communication Mechanism
- Blackboard (partial) — Store holds shared state; only coordinator writes, 
  agents read via prompt context


## References
- https://www.mindstudio.ai/blog/multi-agent-orchestration-patterns