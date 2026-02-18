# Critical Think: Before Action Analysis

You are about to perform a critical action. Before proceeding, you MUST execute the following 6-step metacognitive analysis to ensure quality and prevent common pitfalls.

## Context
**Action Type:** $ACTION_TYPE
**Proposed Action:** $PROPOSED_ACTION
**User Request:** $USER_REQUEST

---

## Step 1: Core Thesis & Confidence Score

**Thesis:** State clearly what you intend to do and why.

**Initial Confidence Score (1-10):** $CONFIDENCE_SCORE

- 1-3: Low confidence, major gaps in understanding
- 4-7: Moderate confidence, some uncertainty or assumptions
- 8-10: High confidence, clear understanding and approach

---

## Step 2: Foundational Analysis

Identify the **top 3 assumptions** underlying your proposed action:

1. **Assumption 1:** $ASSUMPTION_1
   - **Validation:** How can you verify this? $VALIDATION_1

2. **Assumption 2:** $ASSUMPTION_2
   - **Validation:** How can you verify this? $VALIDATION_2

3. **Assumption 3:** $ASSUMPTION_3
   - **Validation:** How can you verify this? $VALIDATION_3

---

## Step 3: Logical Integrity Check

**Chain of Reasoning:**
- Premise A: $PREMISE_A
- Premise B: $PREMISE_B
- Conclusion: $CONCLUSION

**Logical Fallacies Check:**
- [ ] Confirmation bias (seeking only supporting evidence)
- [ ] Authority bias (blind trust in sources/experts)
- [ ] Halo effect (overvaluing positive attributes)
- [ ] Sunk cost fallacy (continuing due to past investment)

**If any fallacies detected:** $FALLACY_RESPONSE

---

## Step 4: AI-Specific Pitfall Analysis

Check for LLM-specific issues:

**Problem Evasion:**
- [ ] Am I avoiding the actual problem by addressing a simpler version?
- [ ] Am I redefining the problem to match my capabilities?

**Happy Path Bias:**
- [ ] Am I only considering the success scenario?
- [ ] What could go wrong? $ERROR_SCENARIOS

**Over-Engineering:**
- [ ] Am I adding unnecessary complexity?
- [ ] What is the MINIMUM viable solution? $MINIMUM_SOLUTION

**Hallucination Risk:**
- [ ] Am I making claims without verification?
- [ ] Do I need to check documentation or code? $VERIFICATION_NEEDS

---

## Step 5: Risk & Mitigation

**Identified Risks:**

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

---

## Step 6: Synthesis & Revised Recommendation

**Based on the analysis above:**

**Revised Confidence Score (1-10):** $REVISED_CONFIDENCE

**Revised Action Plan:**
$REVISED_PLAN

**Key Adjustments from Original:**
$KEY_ADJUSTMENTS

**Proceed with Action?** (YES/NO/CONDITIONAL)

If CONDITIONAL, specify conditions: $CONDITIONS

---

## Output Format

Provide your analysis in the following structure:

```
## Critical Think: Before [Action Name]

### Step 1: Core Thesis & Confidence
**Thesis:** [Your thesis]
**Initial Confidence:** [X/10]

### Step 2: Foundational Assumptions
1. [Assumption] - [Validation method]
2. [Assumption] - [Validation method]
3. [Assumption] - [Validation method]

### Step 3: Logical Integrity
**Reasoning Chain:** [Your logic]
**Fallacies Detected:** [None or list with response]

### Step 4: AI Pitfalls
**Problem Evasion:** [Check results]
**Happy Path Bias:** [Check results]
**Over-Engineering:** [Check results]
**Hallucination Risk:** [Check results]

### Step 5: Risk Analysis
1. [Risk] - [Mitigation]
2. [Risk] - [Mitigation]
3. [Risk] - [Mitigation]

### Step 6: Synthesis
**Revised Confidence:** [X/10]
**Revised Plan:** [Your revised plan]
**Proceed:** [YES/NO/CONDITIONAL]
```
