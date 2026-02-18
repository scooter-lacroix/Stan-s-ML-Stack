# Critical Think: After Action Validation

You have completed an action. Now you MUST validate the quality and correctness of your work using the following 6-step metacognitive framework.

## Context
**Action Type:** $ACTION_TYPE
**Action Completed:** $COMPLETED_ACTION
**User Request:** $USER_REQUEST

---

## Step 1: Core Validation & Confidence Score

**What Was Delivered:** $DELIVERABLE_SUMMARY

**Does it match the user's request?**
- [ ] Yes, completely
- [ ] Partially (gaps: $GAPS)
- [ ] No (missed: $MISSED_ITEMS)

**Actual Confidence Score (1-10):** $ACTUAL_CONFIDENCE

- 1-3: Major issues or gaps
- 4-7: Adequate but with room for improvement
- 8-10: High quality, ready for use

---

## Step 2: Foundational Validation

**Verify the assumptions you made before the action:**

1. **Assumption 1:** $ASSUMPTION_1
   - **Status:** Validated / Invalidated / Uncertain
   - **Evidence:** $EVIDENCE_1
   - **Correction Needed:** $CORRECTION_1

2. **Assumption 2:** $ASSUMPTION_2
   - **Status:** Validated / Invalidated / Uncertain
   - **Evidence:** $EVIDENCE_2
   - **Correction Needed:** $CORRECTION_2

3. **Assumption 3:** $ASSUMPTION_3
   - **Status:** Validated / Invalidated / Uncertain
   - **Evidence:** $EVIDENCE_3
   - **Correction Needed:** $CORRECTION_3

---

## Step 3: Logical Integrity Verification

**Was the reasoning sound?**
- [ ] Yes, conclusion follows from premises
- [ ] No, logical gap detected: $LOGICAL_GAP

**Were there fallacies in execution?**
- [ ] Confirmation bias (only looked for confirming evidence)
- [ ] Authority bias (trusted sources without verification)
- [ ] Other: $OTHER_FALLACY

**If issues found:** $CORRECTION_ACTION

---

## Step 4: Quality & Pitfall Check

**Problem Evasion Check:**
- [ ] I addressed the actual problem, not a simplified version
- [ ] If not: $EVASION_ISSUE

**Happy Path Check:**
- [ ] I considered and handled error cases
- [ ] Unhandled scenarios: $UNHANDLED_SCENARIOS

**Over-Engineering Check:**
- [ ] The solution is appropriately scoped
- [ ] If over-engineered: $OVER_ENGINEERING_ISSUE

**Hallucination Check:**
- [ ] All claims are verified and accurate
- [ ] Unverified claims: $UNVERIFIED_CLAIMS

**Code/Content Quality:**
- [ ] No obvious bugs or errors
- [ ] Follows best practices
- [ ] Maintainable and clear
- [ ] Issues found: $QUALITY_ISSUES

---

## Step 5: Outcome Assessment

**Expected vs Actual:**

| Aspect | Expected | Actual | Gap |
|--------|----------|--------|-----|
| Functionality | $EXPECTED_FUNC | $ACTUAL_FUNC | $FUNC_GAP |
| Quality | $EXPECTED_QUALITY | $ACTUAL_QUALITY | $QUALITY_GAP |
| Completeness | $EXPECTED_COMPLETE | $ACTUAL_COMPLETE | $COMPLETE_GAP |

**Risks Materialized:**
1. $RISK_1_STATUS
2. $RISK_2_STATUS
3. $RISK_3_STATUS

**New Issues Discovered:**
$NEW_ISSUES

---

## Step 6: Synthesis & Recommendations

**Overall Assessment:**
$OVERALL_ASSESSMENT

**Confidence Change:** $BEFORE_CONFIDENCE → $AFTER_CONFIDENCE

**Immediate Corrections Needed:**
$IMMEDIATE_CORRECTIONS

**Future Improvements:**
$FUTURE_IMPROVEMENTS

**Lessons Learned:**
$LESSONS_LEARNED

**Ready to Deliver?** (YES/NO/CONDITIONAL)

If CONDITIONAL, what remains: $REMAINING_WORK

---

## Output Format

Provide your validation in the following structure:

```
## Critical Think: After [Action Name]

### Step 1: Core Validation
**Delivered:** [Summary]
**Matches Request:** [Yes/Partially/No]
**Actual Confidence:** [X/10]

### Step 2: Assumption Validation
1. [Assumption] - [Status] - [Evidence/Correction]
2. [Assumption] - [Status] - [Evidence/Correction]
3. [Assumption] - [Status] - [Evidence/Correction]

### Step 3: Logical Verification
**Reasoning:** [Sound/Flawed]
**Fallacies:** [None or list]

### Step 4: Quality Check
**Problem Evasion:** [Pass/Fail]
**Happy Path:** [Pass/Fail - unhandled cases]
**Over-Engineering:** [Pass/Fail]
**Hallucination:** [Pass/Fail]
**Quality Issues:** [List or None]

### Step 5: Outcome Assessment
**Expected vs Actual:** [Table or summary]
**Risks Materialized:** [Status]
**New Issues:** [List or None]

### Step 6: Synthesis
**Assessment:** [Overall quality judgment]
**Confidence Change:** [X → Y]
**Corrections Needed:** [Immediate fixes]
**Ready to Deliver:** [YES/NO/CONDITIONAL]
```
