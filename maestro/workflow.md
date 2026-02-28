# Project Workflow

## Guiding Principles

1. **The Plan is the Source of Truth:** All work must be tracked in `plan.md`
2. **The Tech Stack is Deliberate:** Changes to the tech stack must be documented in `tech-stack.md` *before* implementation
3. **Test-Driven Development:** Write unit tests before implementing functionality
4. **High Code Coverage:** Aim for >95% code coverage for all modules
5. **User Experience First:** Every decision should prioritize user experience
6. **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use `CI=true` for watch-mode tools (tests, linters) to ensure single execution.
7. **Commit Strategy:** Stage changes aggressively. Do not commit after every small task. Accumulate changes into the largest possible logical commits. When the staging area is significantly large or a logical phase is complete, notify the user to review and confirm the commit. Commit messages must be descriptive but concise.
8. **Task Documentation:** Use git notes for detailed task summaries and keep commit messages concise.

## Agent Usage Requirements

**CRITICAL SYSTEM DIRECTIVE: PROACTIVE AUTOMATIC AGENT USAGE**

1. **Automatic Agent Selection is MANDATORY:** Agent usage is the DEFAULT behavior, NOT an optional enhancement. You MUST proactively select and launch agents based on task complexity WITHOUT waiting for user instruction. The user has configured Maestro to use agents automatically.

2. **NEVER Ask for Agent Permission:** Do NOT ask "Should I use an agent?" or "Which agent should I use?". Make the decision automatically based on task complexity and launch the appropriate agents. Agent selection is YOUR responsibility, not the user's.

3. **Agent Selection Criteria (Execute Automatically):**
   - **Trivial tasks (1-5 lines, simple changes):** Implement directly
   - **Standard tasks (5-50 lines, single file):** Automatically use appropriate implementation agents (opencode-scaffolder)
   - **Complex tasks (multiple files, >50 lines):** Automatically use sonnet-specialist or gemini-analyzer for design + appropriate implementation agents
   - **Large codebase analysis (>100KB):** Automatically use gemini-analyzer for exploration
   - **Spec-driven/ambiguous requirements:** Automatically use droid-factory for specification
   - **ALL implementation work:** MUST be automatically followed by codex-reviewer for validation

**Core Agents:**
- **codex-reviewer**(alias=oracle): Architecture, code review, strategy. (MANDATORY for all implementation)
- **gemini-analyzer**(alias=librarian): Multi-repo analysis, doc lookup, implementation examples.
- **opencode-scaffolder**(alias=macgyver): Fast codebase exploration and pattern matching.
- **gemini-frontend-designer**(alias=michaelangello): Designer turned developer. Builds gorgeous UIs.
- **sonnet-specialist**(alias=hobbs): Technical writing expert. Writes prose that flows.
- **general-purpose**(alias=luis): Visual content specialist. Analyzes PDFs, images, diagrams.
- **droid-factory**(alias=dexter): Specialized in spec-driven development and strategic planning.

**Code Implementation Agents:**
- **qwen-coder**, **kilocode-orchestrator**: Trivial tasks even those exceeding 1-5 lines
- **amp-code**, **opencode-scaffolder**: Standard and some complex tasks
- **rovo-dev**, **droid-factory**: Complex tasks that require pragmatic, high quality code implementations and sound, consistent logic
- **opus-specialist**(alias=einstein): Tasks that involve complex algorithms, complicated, multi-layered logic and reasoning and exceptional coding capabilities with thorough edge case consideration.

**Orchestrator Agents:**
- **kilocode-orchestrator**(alias=kant): Large-scale projects with persistent memory across sessions.
- **llm-council-evaluator**(alias=legion): Meta-agent selection for high-risk or complex decisions.

4. **Mandatory Pre-Commit Review:** Before marking any task complete and committing:
   - ALL code changes MUST be automatically reviewed by codex-reviewer agent
   - Review results MUST be addressed before proceeding
   - If critical issues are found, they MUST be fixed before commit

5. **Quota Awareness:**
   - gemini-analyzer: 300 requests/day (use sparingly for large analysis)
   - gemini-frontend-designer: Unlimited free (use liberally for prototyping)
   - sonnet-specialist: Separate credit pool (use to preserve main quotas)
   - opencode-scaffolder: 2000 requests/day (use for standard implementation)

## Agent Availability and Fallbacks

### Checking Tool Availability

Before using specialized agents that require external CLI tools, you MUST check if the required tools are available on the user's system.

**Check Command:**
```bash
# Check if perspective CLI tools are available
which gemini-cli 2>/dev/null && which qwen-cli 2>/dev/null
```

### Graceful Degradation

If an external agent is unavailable but recommended:
1. Inform the user which agent is unavailable
2. Provide the recommended alternative
3. Ask if they want to proceed with the alternative
4. Continue work with user's approval

## Task Workflow

All tasks follow a strict lifecycle:

### Standard Task Workflow

**AUTOMATIC AGENT SELECTION:**
- You MUST assess task complexity and launch appropriate agents automatically
- Do NOT await user instruction - agent usage is automatic and proactive
- See "Agent Selection Criteria" above for specific guidance

1. **Select Task:** Choose the next available task from `plan.md` in sequential order

2. **Mark In Progress:** Before beginning work, edit `plan.md` and change the task from `[ ]` to `[~]`

3. **Assess Complexity and Select Agent (AUTOMATIC):**
   - **CRITICAL:** Assess task complexity and automatically select the appropriate approach:
     - Trivial tasks (1-5 lines): Implement directly using qwen-coder
     - Standard tasks (5-50 lines, single file): Automatically launch opencode-scaffolder for implementation
     - Complex tasks (multiple files, >50 lines): Automatically launch codex-reviewer or gemini-analyzer for design + opencode-scaffolder for implementation
   - **Do NOT ask user permission** - this is automatic

4. **Write Failing Tests (Red Phase):**
   - Create a new test file for the feature or bug fix.
   - Write one or more unit tests that clearly define the expected behavior and acceptance criteria for the task.
   - **CRITICAL:** Run the tests and confirm that they fail as expected. This is the "Red" phase of TDD. Do not proceed until you have failing tests.
   - **AUTOMATIC AGENT:** For test writing, automatically use appropriate agent (opencode-scaffolder) for standard/complex tasks

5. **Implement to Pass Tests (Green Phase):**
   - Write the minimum amount of application code necessary to make the failing tests pass.
   - Run the test suite again and confirm that all tests now pass. This is the "Green" phase.
   - **AUTOMATIC AGENT:** For implementation, automatically use appropriate agent (opencode-scaffolder) or codex-reviewer/gemini-analyzer + opencode-scaffolder (complex)

6. **Refactor (Optional but Recommended):**
   - With the safety of passing tests, refactor the implementation code and the test code to improve clarity, remove duplication, and enhance performance without changing the external behavior.
   - Rerun tests to ensure they still pass after refactoring.
   - **AUTOMATIC AGENT:** Automatically use opencode-scaffolder for refactoring

7. **Verify Coverage:** Run coverage reports using the project's chosen tools. For example, in a Python project, this might look like:
   ```bash
   pytest --cov=app --cov-report=html
   ```
   Target: >95% coverage for new code. The specific tools and commands will vary by language and framework.

8. **Document Deviations:** If implementation differs from tech stack:
   - **STOP** implementation
   - Update `tech-stack.md` with new design
   - Add dated note explaining the change
   - Resume implementation

9. **Agent Review (MANDATORY - AUTOMATIC):**
   **CRITICAL:** Before proceeding to commit, you MUST automatically launch codex-reviewer. Do NOT wait for user instruction.
   - **AUTOMATICALLY Launch Code Review:** Use the codex-reviewer agent to review all changes made during this task. Provide context: task description, files changed, expected outcomes.
   - **Address Review Findings:** If critical issues are found, fix them before proceeding. If suggestions are provided, address or document decision to defer.
   - **Confirm Review Complete:** Only after codex-reviewer passes should you proceed to commit. Document any issues found and resolved.

10. **Commit Code Changes:**
    - Stage all code changes related to the task.
    - Propose a clear, concise commit message e.g, `feat(ui): Create basic HTML structure for calculator`.
    - Perform the commit.

11. **Attach Task Summary with Git Notes:**
    - **Step 11.1: Get Commit Hash:** Obtain the hash of the *just-completed commit* (`git log -1 --format="%H"`).
    - **Step 11.2: Draft Note Content:** Create a detailed summary for the completed task. This should include the task name, a summary of changes, a list of all created/modified files, and the core "why" for the change.
    - **Step 11.3: Attach Note:** Use the `git notes` command to attach the summary to the commit.
      ```bash
      # The note content from the previous step is passed via the -m flag.
      git notes add -m "<note content>" <commit_hash>
      ```

12. **Get and Record Task Commit SHA:**
    - **Step 12.1: Update Plan:** Read `plan.md`, find the line for the completed task, update its status from `[~]` to `[x]`, and append the first 7 characters of the *just-completed commit's* commit hash.
    - **Step 12.2: Write Plan:** Write the updated content back to `plan.md`.

13. **Commit Plan Update:**
    - **Action:** Stage the modified `plan.md` file.
    - **Action:** Commit this change with a descriptive message (e.g., `maestro(plan): Mark task 'Create user model' as complete`).

## Phase Completion Verification and Checkpointing Protocol

**MODE CONFIGURATION:** The workflow mode is configured during project setup in `maestro/workflow-config.json`.

**Trigger:** This protocol is executed immediately after a task is completed that also concludes a phase in `plan.md` until the track being worked on is completed.

**IMPORTANT:** The workflow behavior depends on the configured mode:

### 🔄 Autonomous Mode

When `workflow_mode` is set to `"autonomous"` with `checkpoint_interval`:

- **Automatic phase verification** after each phase completion
- **No user confirmation required** until the configured checkpoint interval
- **Immediate progression** to next phase after checkpointing
- **User checkpoints** occur at the configured interval (e.g., every 3rd phase, every quarter, every half)

### 🛑 Manual User Verification (Final Phase - All Modes)

**MANDATORY:** When the **final phase** of the track is completed, ALL workflow modes **pause** and require explicit user verification. The user must review the complete implementation, including all code changes, test results, and documentation. Only after the user provides explicit approval to proceed should the checkpoint commit be created and the track marked as complete.

### 📋 8-Step Verification Process (Autonomous Mode)

Execute this process after each phase completion (pausing only at configured checkpoints):

1. **Announce Protocol Start:** Inform the user that the phase is complete and the verification and checkpointing protocol has begun.

2. **Ensure Test Coverage for Phase Changes:**
   - **Step 2.1: Determine Phase Scope:** To identify the files changed in this phase, you must first find the starting point. Read `plan.md` to find the Git commit SHA of the *previous* phase's checkpoint. If no previous checkpoint exists, the scope is all changes since the first commit.
   - **Step 2.2: List Changed Files:** Execute `git diff --name-only <previous_checkpoint_sha> HEAD` to get a precise list of all files modified during this phase.
   - **Step 2.3: Verify and Create Tests:** For each file in the list:
     - **CRITICAL:** First, check its extension. Exclude non-code files (e.g., `.json`, `.md`, `.yaml`, `.toml`, `.txt`, `.sh`).
     - For each remaining code file, verify a corresponding test file exists.
     - If a test file is missing, you **must** create one. Before writing the test, **first, analyze other test files in the repository to determine the correct naming convention and testing style.** The new tests **must** validate the functionality described in this phase's tasks (`plan.md`).

3. **Execute Automated Tests with Proactive Debugging:**
   - Before execution, announce the exact shell command you will use to run the tests.
   - **Example Announcement:** "I will now run the automated test suite to verify the phase. **Command:** `CI=true cargo test`"
   - Execute the announced command.
   - **AUTONOMOUS DEBUGGING:** If tests fail, you **must** attempt to fix them automatically. You may attempt to fix failures up to **5 times**. Use appropriate subagents to diagnose and fix issues. If tests still fail after 5 attempts, **halt execution** and report the persistent failure with full diagnostic information.

4. **Conduct "Tzar of Excellence" Review (MANDATORY):**
   - **CRITICAL:** Before creating the checkpoint commit, you MUST conduct a rigorous zero-tolerance code review using the configured review agent.
   - **Deploy Review Agent:** Invoke the configured review agent (from `maestro/workflow-config.json`) with the "Tzar of Excellence" directive (see below).
   - **Wait for review completion:** Do not proceed until the review is complete.
   - **Address ALL critical findings:** Fix any critical issues identified by the review.
   - **Re-test if needed:** If fixes were made, re-run tests to ensure nothing broke.
   - **Document review:** Create a review summary document at `docs/phase-[X]-tzar-review.md`.
   - **Only then proceed:** Once the review passes with no critical issues, continue to step 5.

   **"Tzar of Excellence" Directive:**
   ```
   You are conducting the "Tzar of Excellence" review for Phase [X] of this track.

   ## Zero Tolerance Excellence Directive

   You are reviewing a completed phase with ZERO tolerance for:
   - Mediocrity
   - Corner cases unhandled
   - Missing error handling
   - Security vulnerabilities
   - Poor performance
   - Incomplete implementations
   - Technical debt
   - Code quality issues

   ## Review Scope

   Review ALL code changes made during this phase:
   - All commits in this phase
   - All files created/modified
   - All implementations
   - All tests
   - Edge cases covered?

   ## Required Assessments

   1. **Code Quality** - Is the code production-ready? Maintainable? Optimized?
   2. **Logic & Correctness** - Is logic sound? Edge cases handled? Error handling comprehensive?
   3. **Security** - Any vulnerabilities? Input validation complete? Injection risks?
   4. **Performance** - Bottlenecks? Optimizations needed? Unnecessary operations?
   5. **Comprehensive Nature** - All edge cases covered? Error handling complete? Implementation complete?

   ## Required Output

   1. Critical Issues List (must fix before proceeding)
   2. Improvements Needed (should fix for excellence)
   3. Optimization Opportunities
   4. Edge Cases Not Handled
   5. Security Concerns
   6. Performance Issues
   7. Final Verdict: PASS/FAIL with detailed reasoning

   ## Zero Tolerance Means

   - No "good enough" - must be excellent
   - No "it works" - must be robust
   - No "later" - must be complete now
   - No "maybe" - must be certain

   Be brutal. Be thorough. Be excellent.
   ```

   **Failure Criteria:** Phase review FAILS if any critical security vulnerabilities, unhandled edge cases, missing error handling, performance issues, incomplete implementations, or blocking technical debt are found.

5. **Create Checkpoint Commit:**
   - Stage all changes. If no changes occurred in this step, proceed with an empty commit.
   - Perform the commit with a clear and concise message (e.g., `maestro(checkpoint): Checkpoint end of Phase X`).

6. **Attach Verification Report using Git Notes:**
   - **Step 6.1: Draft Note Content:** Create a detailed verification report including the automated test command, test results, and any issues resolved.
   - **Step 6.2: Attach Note:** Use the `git notes` command and the full commit hash from the previous step to attach the full report to the checkpoint commit.

7. **Get and Record Phase Checkpoint SHA:**
   - **Step 7.1: Get Commit Hash:** Obtain the hash of the *just-created checkpoint commit* (`git log -1 --format="%H"`).
   - **Step 7.2: Update Plan:** Read `plan.md`, find the heading for the completed phase, and append the first 7 characters of the commit hash in the format `[checkpoint: <sha>]`.
   - **Step 7.3: Write Plan:** Write the updated content back to `plan.md`.

8. **Commit Plan Update:**
   - **Action:** Stage the modified `plan.md` file.
   - **Action:** Commit this change with a descriptive message following the format `maestro(plan): Mark phase '<PHASE NAME>' as complete`.

9. **Checkpoint Check (Autonomous Mode Only):**
   - **Read workflow config:** Check `maestro/workflow-config.json` for the current checkpoint interval.
   - **Calculate progress:** Determine if this checkpoint corresponds to a configured interval (e.g., 33%, 50%, 75%, etc.).
   - **If checkpoint interval reached:** Pause and await user verification before proceeding to the next phase.
   - **If checkpoint interval not reached:** Immediately proceed to the next phase without waiting for confirmation.

10. **Announce Completion and Continue:** Inform the user that the phase is complete and the checkpoint has been created. Then:
   - **Autonomous Mode:** If checkpoint interval not reached, immediately proceed to the next phase without waiting for confirmation.
   - **Checkpoint Reached/Final Phase:** Pause for user verification before proceeding.

### Quality Gates Checklist

Before marking any task or phase complete, verify:

- [ ] All tests pass
- [ ] 95%+ code coverage
- [ ] Code style compliance
- [ ] Documentation completeness
- [ ] Type safety enforcement
- [ ] No linting/static analysis errors
- [ ] Updated documentation
- [ ] No security vulnerabilities

## Development Commands

**AI AGENT INSTRUCTION: This section should be adapted to the project's specific language, framework, and build tools.**

### Setup
```bash
# Example: Commands to set up the development environment (e.g., install dependencies, configure database)
# e.g., for a Node.js project: npm install
# e.g., for a Go project: go mod tidy
```

### Daily Development
```bash
# Example: Commands for common daily tasks (e.g., start dev server, run tests, lint, format)
# e.g., for a Node.js project: npm run dev, npm test, npm run lint
# e.g., for a Go project: go run main.go, go test ./..., go fmt ./...
```

### Before Committing
```bash
# Example: Commands to run all pre-commit checks (e.g., format, lint, type check, run tests)
# e.g., for a Node.js project: npm run check
# e.g., for a Go project: make check (if a Makefile exists)
```

## Testing Requirements

### Unit Testing
- Every module must have corresponding tests.
- Use appropriate test setup/teardown mechanisms (e.g., fixtures, beforeEach/afterEach).
- Mock external dependencies.
- Test both success and failure cases.

### Integration Testing
- Test complete user flows
- Verify database transactions
- Test authentication and authorization
- Check form submissions

## Code Review Process

### Self-Review Checklist
Before requesting review:

1. **Functionality**
   - Feature works as specified
   - Edge cases handled
   - Error messages are user-friendly

2. **Code Quality**
   - Follows style guide
   - DRY principle applied
   - Clear variable/function names
   - Appropriate comments

3. **Testing**
   - Unit tests comprehensive
   - Integration tests pass
   - Coverage adequate (>95%)

4. **Security**
   - No hardcoded secrets
   - Input validation present
   - SQL injection prevented
   - XSS protection in place

5. **Performance**
   - Database queries optimized
   - Images optimized
   - Caching implemented where needed

## Commit Guidelines

### Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks

### Examples
```bash
git commit -m "feat(auth): Add remember me functionality"
git commit -m "fix(posts): Correct excerpt generation for short posts"
git commit -m "test(comments): Add tests for emoji reaction limits"
git commit -m "style(mobile): Improve button touch targets"
```

## Definition of Done

A task is complete when:

1. All code implemented to specification
2. Unit tests written and passing
3. Code coverage meets project requirements (>95%)
4. Documentation complete (if applicable)
5. Code passes all configured linting and static analysis checks
6. Code reviewed by review agent with all critical issues addressed
7. Implementation notes added to `plan.md`
8. Changes committed with proper message
9. Git note with task summary attached to the commit

## Emergency Procedures

### Critical Bug in Production
1. Create hotfix branch from main
2. Write failing test for bug
3. Implement minimal fix
4. Test thoroughly
5. Deploy immediately
6. Document in plan.md

### Data Loss
1. Stop all write operations
2. Restore from latest backup
3. Verify data integrity
4. Document incident
5. Update backup procedures

### Security Breach
1. Rotate all secrets immediately
2. Review access logs
3. Patch vulnerability
4. Notify affected users (if any)
5. Document and update security procedures

## Deployment Workflow

### Pre-Deployment Checklist
- [ ] All tests passing
- [ ] Coverage >95%
- [ ] No linting errors
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] Backup created

### Deployment Steps
1. Merge feature branch to main
2. Tag release with version
3. Push to deployment service
4. Run database migrations
5. Verify deployment
6. Test critical paths
7. Monitor for errors

### Post-Deployment
1. Monitor analytics
2. Check error logs
3. Gather user feedback
4. Plan next iteration

## Continuous Improvement

- Review workflow weekly
- Update based on pain points
- Document lessons learned
- Optimize for user happiness
- Keep things simple and maintainable
