# Critical Think: Q&A Phase

You are about to ask the user a clarifying question. Execute this 6-step analysis to ensure the question is necessary, well-formed, and won't introduce bias.

## Context
**Current Task:** $CURRENT_TASK
**Intended Question:** $INTENDED_QUESTION
**Question Purpose:** $QUESTION_PURPOSE

---

## Step 1: Core Thesis & Confidence

**Why I need to ask this:**
$NEED_TO_ASK_REASONING

**Is this question NECESSARY?**
- [ ] Yes, critical information is missing
- [ ] Maybe, could infer with reasonable confidence
- [ ] No, I have enough information to proceed

**Confidence that question is necessary (1-10):** $QUESTION_NECESSITY_CONFIDENCE

**If confidence < 7:** Consider whether you can proceed without asking.

---

## Step 2: Foundational Analysis

**What assumptions am I making that lead to this question?**

1. **Assumption 1:** $ASSUMPTION_1
   - **Could I verify this instead?** $VERIFICATION_OPTION_1

2. **Assumption 2:** $ASSUMPTION_2
   - **Could I verify this instead?** $VERIFICATION_OPTION_2

3. **Assumption 3:** $ASSUMPTION_3
   - **Could I verify this instead?** $VERIFICATION_OPTION_3

---

## Step 3: Logical Integrity Check

**Question Clarity:**
- [ ] The question is unambiguous
- [ ] The question is specific enough to be answerable
- [ ] The question is not leading (doesn't suggest an answer)

**Potential Issues:**
- [ ] Leading question (suggests desired answer)
- [ ] Multiple questions in one
- [ ] Ambiguous or vague
- [ ] Assumes false premise

**If issues detected:** $QUESTION_REFORMULATION

---

## Step 4: AI-Specific Pitfall Check

**Authority Bias Check:**
- [ ] Am I asking because I lack confidence, not because I actually need clarification?
- [ ] Am I deferring to the user when I could make a reasonable decision?

**Problem Evasion Check:**
- [ ] Am I asking questions to avoid doing the work of making decisions?
- [ ] Am I asking questions I should be able to answer myself?

**Over-Questioning Risk:**
- [ ] Is this question essential or nice-to-have?
- [ ] Can I provide a recommendation with the question instead of just asking?

**User Experience:**
- [ ] Will this question feel like stalling?
- [ ] Can I provide context/options to make answering easier?

---

## Step 5: Risk & Mitigation

**Risks of asking this question:**

1. **Risk:** $RISK_1
   - **Mitigation:** $MITIGATION_1

2. **Risk:** $RISK_2
   - **Mitigation:** $MITIGATION_2

**Risks of NOT asking this question:**

1. **Risk:** $RISK_OF_NOT_ASKING_1
2. **Risk:** $RISK_OF_NOT_ASKING_2

**Risk Assessment:**
- Asking is riskier: $ASKING_RISKIER_REASON
- Not asking is riskier: $NOT_ASKING_RISKIER_REASON

---

## Step 6: Synthesis & Decision

**Analysis Summary:**
$ANALYSIS_SUMMARY

**Decision:**
- [ ] PROCEED with asking the question
- [ ] SKIP the question (proceed with reasonable assumption)
- [ ] REFINE the question first

**If SKIP:** What assumption will you use? $ASSUMPTION_TO_USE

**If REFINE:** Revised question: $REVISED_QUESTION

**If PROCEED:** Final question to ask: $FINAL_QUESTION

**Should I provide recommendations/options with the question?**
$RECOMMENDATIONS_WITH_QUESTION

---

## Output Format

```
## Critical Think: Before Question

### Step 1: Core Thesis
**Why Ask:** [Reasoning]
**Necessary:** [Yes/Maybe/No]
**Confidence:** [X/10]

### Step 2: Assumptions
1. [Assumption] - [Could verify?]
2. [Assumption] - [Could verify?]
3. [Assumption] - [Could verify?]

### Step 3: Question Quality
**Clear:** [Yes/No]
**Specific:** [Yes/No]
**Non-leading:** [Yes/No]
**Issues:** [None or list]

### Step 4: AI Pitfalls
**Authority Bias:** [Check result]
**Problem Evasion:** [Check result]
**Over-Questioning:** [Check result]
**User Experience:** [Assessment]

### Step 5: Risk Analysis
**Risks of Asking:** [List]
**Risks of Not Asking:** [List]
**Risk Assessment:** [Which is riskier]

### Step 6: Decision
**Decision:** [PROCEED/SKIP/REFINE]
**Final Question:** [The question]
**Include Options:** [Yes/No - what options]
```
