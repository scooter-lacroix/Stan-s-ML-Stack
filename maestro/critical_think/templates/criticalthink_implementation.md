# Critical Think: Code Implementation

You are about to implement code. Execute this 6-step analysis to ensure the implementation will be correct, robust, and maintainable.

## Context
**Implementation Task:** $TASK
**Files to Modify:** $FILES
**Tech Stack:** $TECH_STACK
**Requirements:** $REQUIREMENTS

---

## Step 1: Core Thesis & Confidence

**What I will implement:**
$IMPLEMENTATION_THESIS

**How it will work:**
$IMPLEMENTATION_APPROACH

**Initial confidence in approach (1-10):** $INITIAL_CONFIDENCE

---

## Step 2: Foundational Analysis

**What assumptions am I making about this implementation?**

1. **Assumption 1:** $ASSUMPTION_1
   - **Verification:** $VERIFICATION_1

2. **Assumption 2:** $ASSUMPTION_2
   - **Verification:** $VERIFICATION_2

3. **Assumption 3:** $ASSUMPTION_3
   - **Verification:** $VERIFICATION_3

**Dependencies:**
$DEPENDENCIES

**Constraints:**
$CONSTRAINTS

---

## Step 3: Logical Integrity Check

**Design Verification:**
- [ ] Follows existing architectural patterns
- [ ] Integrates properly with existing code
- [ ] Maintains consistency with codebase style
- [ ] Respects the principle of least surprise

**Logic Flow:**
$LOGIC_FLOW

**Potential Logic Issues:**
$LOGIC_ISSUES

**Edge Cases Identified:**
$EDGE_CASES

---

## Step 4: AI-Specific Pitfall Check

**Problem Evasion:**
- [ ] I'm solving the actual problem, not a simplified version
- [ ] The solution addresses all requirements
- [ ] No critical features omitted

**Happy Path Bias:**
- [ ] Error handling included
- [ ] Input validation planned
- [ ] Failure modes considered
- [ ] Graceful degradation designed

**Over-Engineering:**
- [ ] Solution is appropriately scoped
- [ ] YAGNI principle followed
- [ ] No unnecessary abstractions
- [ ] Minimal viable implementation

**Hallucination Risk:**
- [ ] I will verify API usage against docs
- [ ] I will test code before considering complete
- [ ] I will check existing code before making assumptions
- [ ] No unverified language/framework features used

---

## Step 5: Risk & Mitigation

**Implementation Risks:**

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

**Test Coverage Plan:**
$TEST_PLAN

**Rollback Strategy (if needed):**
$ROLLBACK_STRATEGY

---

## Step 6: Synthesis & Implementation Plan

**Implementation Steps:**
$IMPLEMENTATION_STEPS

**Testing Strategy:**
$TESTING_STRATEGY

**Quality Checks:**
$QUALITY_CHECKS

**Revised Confidence (1-10):** $REVISED_CONFIDENCE

**Proceed with Implementation?** (YES/NO/CONDITIONAL)

---

## Output Format

```
## Critical Think: Before Implementation

### Step 1: Core Thesis
**What to Implement:** [Thesis]
**How it Works:** [Approach]
**Initial Confidence:** [X/10]

### Step 2: Assumptions
1. [Assumption] - [Verification]
2. [Assumption] - [Verification]
3. [Assumption] - [Verification]

**Dependencies:** [List]
**Constraints:** [List]

### Step 3: Logical Integrity
**Design:** [Verification checks]
**Logic Flow:** [Description]
**Edge Cases:** [List]

### Step 4: AI Pitfalls
**Problem Evasion:** [Check]
**Happy Path:** [Check + error handling]
**Over-Engineering:** [Check]
**Hallucination:** [Check + verification plan]

### Step 5: Risk Analysis
1. [Risk] - [Mitigation]
2. [Risk] - [Mitigation]
3. [Risk] - [Mitigation]

**Test Plan:** [Coverage strategy]
**Rollback:** [Strategy if needed]

### Step 6: Synthesis
**Implementation Steps:** [Numbered list]
**Testing Strategy:** [Approach]
**Quality Checks:** [List]
**Revised Confidence:** [X/10]
**Proceed:** [YES/NO/CONDITIONAL]
```

---

## After Implementation

After implementing code, run the **After Action** validation:

1. **Code Review Checklist:**
   - [ ] Code follows project style guidelines
   - [ ] No hardcoded values (use constants/config)
   - [ ] No commented-out code
   - [ ] No debug statements
   - [ ] Proper error handling
   - [ ] Input validation

2. **Testing Checklist:**
   - [ ] All tests pass
   - [ ] Edge cases covered
   - [ ] Error cases tested
   - [ ] Coverage >98%

3. **Integration Checklist:**
   - [ ] Works with existing code
   - [ ] No breaking changes
   - [ ] API contracts maintained
   - [ ] Dependencies correct

4. **Documentation:**
   - [ ] Code is self-documenting
   - [ ] Complex logic has comments
   - [ ] Public API documented
