# Rusty Stack Master Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the Rusty Stack update, upgrade, Rust migration, Windows foundation, and telemetry programs in a blocked sequence that preserves validated behavior and only permits parallel execution when file overlap and interface churn risk are both below 10%.

**Architecture:** One shared Rust contract layer feeds five implementation tracks: update, upgrade, Rust migration, Windows foundation, and telemetry/stability. The master orchestration plan governs milestone order, Tzar gates, handoff updates, and when child plans may run in serial or in low-overlap parallel.

**Tech Stack:** Markdown planning docs, Rust workspace crates under `rusty-stack/`, shell validation tests under `tests/validation/`, git-based review workflow.

---

### Task 1: Freeze The Program Control Files

**Parallelism:** Serial only — these docs define the rules for every other task.

**Files:**
- Modify: `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`
- Modify: `docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md`
- Create: `docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md`

- [ ] **Step 1: Add the failing governance review entries**

Append this state block to `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md` only after confirming it is not already present:

```md
## Program Control Contract

- Child implementation plans may not start execution until the written spec set has user approval and a recorded Tzar decision.
- Shared manifest, planner, and validation schema work is serial-only.
- Child plans may run in parallel only when this ledger explicitly marks them parallel-safe.
```

- [ ] **Step 2: Run a diff review so the new governance text is visible**

Run: `git diff -- docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md`
Expected: Diff shows only the new control-language additions.

- [ ] **Step 3: Commit the governance freeze**

```bash
git add docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md
git commit -m "docs(plan): add master orchestration controls"
```

### Task 2: Record The Blocked Milestone Graph

**Parallelism:** Serial only — milestone dependencies are the top-level execution contract.

**Files:**
- Modify: `docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md`
- Test: `docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`

- [ ] **Step 1: Write the failing milestone-order assertion into the plan**

Add this exact checklist section to the plan:

```md
## Blocked Execution Graph

- [ ] Milestone 0: Written specs approved and Tzar-reviewed
- [ ] Milestone 1: Shared core contracts landed and passing
- [ ] Milestone 2: `rusty upgrade` compatibility gate landed and passing
- [ ] Milestone 3: `rusty update` scan/plan flow landed and passing
- [ ] Milestone 4: `rusty update` apply/verify flow landed and passing
- [ ] Milestone 5: Rust migration wave 1 landed and passing
- [ ] Milestone 6: Telemetry/stability landed and passing
- [ ] Milestone 7: Windows foundation landed and passing
```

- [ ] **Step 2: Verify the spec still matches the plan**

Run: `rg -n "Blocked Execution Graph|Milestone 0|Milestone 6" docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`
Expected: Both files mention the same milestone ordering semantics.

- [ ] **Step 3: Commit the milestone graph**

```bash
git add docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md
git commit -m "docs(plan): record blocked milestone graph"
```

### Task 3: Encode Parallelism Gates

**Parallelism:** Serial only — these rules determine whether any later work may run concurrently.

**Files:**
- Modify: `docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md`
- Test: `docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`

- [ ] **Step 1: Add the explicit parallel-safe matrix**

Insert this exact section:

```md
## Parallel-Safe Work Matrix

| Work Item | Parallel Eligible | Reason |
| --- | --- | --- |
| Shared manifest schema | No | Cross-pollution risk exceeds 10% |
| Shared planner contract | No | Cross-pollution risk exceeds 10% |
| Shared validation-tier policy | No | Cross-pollution risk exceeds 10% |
| Windows UX copy and launcher polish after backend contract freeze | Yes | UI-only file overlap below 10% |
| Telemetry ingest docs after payload schema freeze | Yes | Documentation-only overlap below 10% |
| Legacy script retirement for one isolated component after adapter contract freeze | Conditional | Only if file ownership is partitioned and reviewed |
```
```

- [ ] **Step 2: Verify no contradictory language remains**

Run: `rg -n "parallel|Parallel" docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`
Expected: The plan and master spec both describe the same <10% rule with no contradictory exceptions.

- [ ] **Step 3: Commit the parallelism matrix**

```bash
git add docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md
git commit -m "docs(plan): define parallel-safe task matrix"
```

### Task 4: Wire The Tzar Gates Into Execution

**Parallelism:** Serial only — Tzar PASS is the milestone unlock condition.

**Files:**
- Modify: `docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md`
- Modify: `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`
- Modify: `docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md`

- [ ] **Step 1: Add the failing gate checklist to the master plan**

Add this exact section:

```md
## Tzar Gates

- [ ] Gate A: Spec set PASS
- [ ] Gate B: Shared core contracts PASS
- [ ] Gate C: `rusty upgrade` compatibility PASS
- [ ] Gate D: `rusty update` scan/plan PASS
- [ ] Gate E: `rusty update` apply/verify PASS
- [ ] Gate F: Telemetry/stability PASS
- [ ] Gate G: Windows foundation PASS
```

- [ ] **Step 2: Update the handoff file to reflect the next allowed action**

Change the `Next allowed action` line to:

```md
- Next allowed action: Tzar review of the written spec set, then implementation plan execution if PASS is recorded
```

- [ ] **Step 3: Run the documentation consistency check**

Run: `rg -n "Next allowed action|Gate A|Gate F" docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md`
Expected: The same gate vocabulary appears across all three files.

- [ ] **Step 4: Commit the Tzar wiring**

```bash
git add docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md
git commit -m "docs(plan): wire Tzar gates into orchestration"
```

### Task 5: Verify The Control Surface

**Parallelism:** Serial only — this is the completion proof for the orchestration plan.

**Files:**
- Test: `docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md`
- Test: `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`
- Test: `docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md`

- [ ] **Step 1: Run the documentation placeholder scan**

Run: `rg -n "TBD|TODO|FIXME|implement later|fill in details" docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md docs/superpowers/plans --glob '!docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md'`
Expected: No matches.

- [ ] **Step 2: Run the program-control grep check**

Run: `rg -n "Block(ed)? Execution Graph|Parallel-Safe Work Matrix|Tzar Gates|Next allowed action" docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`
Expected: All required control sections are present.

- [ ] **Step 3: Commit the verified orchestration plan**

```bash
git add docs/superpowers/plans/2026-04-24-rusty-stack-master-orchestration-plan.md docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md
git commit -m "docs(plan): finalize orchestration control plan"
```
