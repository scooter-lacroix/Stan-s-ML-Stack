# Critical Think: Agent Delegation

You are about to delegate work to another agent. Execute this 6-step analysis to ensure delegation is appropriate and the handoff will be successful.

## Context
**Current Task:** $CURRENT_TASK
**Proposed Agent:** $PROPOSED_AGENT
**Delegation Reason:** $DELEGATION_REASON
**Work to Delegate:** $WORK_TO_DELEGATE

---

## Step 1: Core Thesis & Confidence

**Why delegate this work:**
$DELEGATION_RATIONALE

**What success looks like:**
$SUCCESS_CRITERIA

**Confidence that delegation is appropriate (1-10):** $DELEGATION_CONFIDENCE

**Confidence in agent capability (1-10):** $AGENT_CAPABILITY_CONFIDENCE

---

## Step 2: Foundational Analysis

**What assumptions am I making about this delegation?**

1. **Assumption 1:** $ASSUMPTION_1
   - **Verification:** $VERIFICATION_1

2. **Assumption 2:** $ASSUMPTION_2
   - **Verification:** $VERIFICATION_2

3. **Assumption 3:** $ASSUMPTION_3
   - **Verification:** $VERIFICATION_3

**Agent Capabilities:**
$AGENT_CAPABILITIES

**Known Agent Limitations:**
$AGENT_LIMITATIONS

---

## Step 3: Logical Integrity Check

**Is delegation the right choice?**
- [ ] Task complexity warrants specialist
- [ ] Agent has the right tools/expertise
- [ ] Task is well-defined enough to delegate
- [ ] Clear acceptance criteria exist

**Alternative approaches considered:**
- [ ] Handle myself: $WHY_NOT_SELF
- [ ] Use different agent: $WHY_NOT_OTHER_AGENT
- [ ] Break down differently: $WHY_NOT_DIFFERENT_BREAKDOWN

**Handoff Quality:**
- [ ] Context provided is sufficient
- [ ] Goal is clear and specific
- [ ] Constraints are documented
- [ ] Success criteria are measurable

**Potential Issues:**
$HANDOFF_ISSUES

---

## Step 4: AI-Specific Pitfall Check

**Authority Bias:**
- [ ] Am I delegating to avoid responsibility?
- [ ] Am I delegating because I lack confidence, not because it's appropriate?
- [ ] Can I provide initial direction before delegating?

**Problem Evasion:**
- [ ] Am I delegating the hard parts of the problem?
- [ ] Am I delegating to avoid making difficult decisions?
- [ ] Is the core problem still mine to solve?

**Over-Delegation Risk:**
- [ ] Is this too trivial to delegate?
- [ ] Will the overhead of delegation exceed the benefit?
- [ ] Can I handle this more efficiently myself?

**Agent Capability Mismatch:**
- [ ] Does the agent actually have the required tools?
- [ ] Is the task within the agent's scope?
- [ ] Am I setting the agent up for failure?

---

## Step 5: Risk & Mitigation

**Delegation Risks:**

1. **Risk:** $RISK_1
   - **Probability:** (Low/Medium/High)
   - **Impact:** (Low/Medium/High)
   - **Mitigation:** $MITIGATION_1

2. **Risk:** $RISK_2
   - **Probability:** (Low/Medium/High)
   - **Impact:** (Low/Medium/High)
   - **Mitigation:** $MITIGATION_2

3. **Risk:** $RISK_3
   - **Probability:** (Low/Medium/High)
   - **Impact:** (Low/Medium/High)
   - **Mitigation:** $MITIGATION_3

**Verification Plan:**
$VERIFICATION_PLAN

**Contingency if delegation fails:**
$CONTINGENCY_PLAN

---

## Step 6: Synthesis & Delegation Plan

**Refined Delegation Decision:**
- [ ] PROCEED with delegation to $AGENT
- [ ] HANDLE myself instead
- [ ] DIFFERENT approach: $ALTERNATIVE

**If proceeding, delegation prompt will include:**
$DELEGATION_PROMPT_ELEMENTS

**Context to provide:**
$CONTEXT_TO_PROVIDE

**Success metrics:**
$SUCCESS_METRICS

**Post-delegation validation:**
$POST_DELEGATION_VALIDATION

**Revised Confidence (1-10):** $REVISED_CONFIDENCE

**Proceed with Delegation?** (YES/NO/CONDITIONAL)

---

## Output Format

```
## Critical Think: Before Agent Delegation

### Step 1: Core Thesis
**Why Delegate:** [Rationale]
**Success Looks Like:** [Criteria]
**Delegation Confidence:** [X/10]
**Agent Capability Confidence:** [X/10]

### Step 2: Assumptions
1. [Assumption] - [Verification]
2. [Assumption] - [Verification]
3. [Assumption] - [Verification]

**Agent Capabilities:** [List]
**Agent Limitations:** [List]

### Step 3: Logical Integrity
**Appropriate Delegation:** [Check results]
**Alternatives Considered:**
- Self: [Why not]
- Other agent: [Why not]
- Different breakdown: [Why not]

**Handoff Quality:** [Checks]

### Step 4: AI Pitfalls
**Authority Bias:** [Check results]
**Problem Evasion:** [Check results]
**Over-Delegation:** [Check results]
**Capability Mismatch:** [Check results]

### Step 5: Risk Analysis
1. [Risk] - [Mitigation]
2. [Risk] - [Mitigation]
3. [Risk] - [Mitigation]

**Verification:** [Plan]
**Contingency:** [Plan if fails]

### Step 6: Synthesis
**Decision:** [PROCEED/HANDLE/ALTERNATIVE]
**Delegation Prompt:** [Key elements]
**Context:** [What to provide]
**Success Metrics:** [Measurable criteria]
**Post-Validation:** [How to verify results]
**Revised Confidence:** [X/10]
**Proceed:** [YES/NO/CONDITIONAL]
```

---

## After Agent Delegation

After the agent returns, run the **After Action** validation:

1. **Result Verification:**
   - [ ] Deliverable matches success criteria
   - [ ] All requirements met
   - [ ] Quality standards met
   - [ ] No obvious errors or issues

2. **Integration Check:**
   - [ ] Work integrates with existing codebase
   - [ ] No breaking changes introduced
   - [ ] Follows project patterns
   - [ ] Documentation is adequate

3. **Quality Validation:**
   - [ ] Code is clean and maintainable
   - [ ] Tests are adequate
   - [ ] Edge cases handled
   - [ ] Error handling present

4. **Agent Performance Assessment:**
   - **Agent strengths observed:** $STRENGTHS
   - **Agent limitations observed:** $LIMITATIONS
   - **Would use this agent again?** $REUSE_DECISION
   - **Lessons for future delegation:** $LESSONS
