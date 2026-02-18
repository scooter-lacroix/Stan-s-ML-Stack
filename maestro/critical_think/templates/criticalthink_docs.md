# Critical Think: Documentation Generation

You are about to generate documentation. Execute this 6-step analysis to ensure the documentation will be accurate, complete, and useful.

## Context
**Documentation Type:** $DOC_TYPE
**Target Audience:** $AUDIENCE
**Subject Matter:** $SUBJECT
**Purpose:** $PURPOSE

---

## Step 1: Core Thesis & Confidence

**What this documentation will convey:**
$DOC_THESIS

**Key information to include:**
$KEY_INFO

**Initial confidence in completeness (1-10):** $INITIAL_CONFIDENCE

---

## Step 2: Foundational Analysis

**What assumptions am I making about the documentation?**

1. **Assumption 1:** $ASSUMPTION_1
   - **Verification:** $VERIFICATION_1

2. **Assumption 2:** $ASSUMPTION_2
   - **Verification:** $VERIFICATION_2

3. **Assumption 3:** $ASSUMPTION_3
   - **Verification:** $VERIFICATION_3

---

## Step 3: Logical Integrity Check

**Documentation Structure:**
- [ ] Introduction (context and purpose)
- [ ] Main content (organized logically)
- [ ] Examples (where applicable)
- [ ] References/links (where applicable)

**Completeness Check:**
- [ ] Prerequisites documented
- [ ] Steps are in logical order
- [ ] All necessary details included
- [ ] Edge cases addressed

**Potential Issues:**
$STRUCTURAL_ISSUES

---

## Step 4: AI-Specific Pitfall Check

**Hallucination Risk:**
- [ ] All claims will be verified against code/docs
- [ ] No assumptions presented as facts
- [ ] Verification plan: $VERIFICATION_PLAN

**Happy Path Bias:**
- [ ] Error scenarios documented
- [ ] Common mistakes included
- [ ] Troubleshooting section included

**Over-Documentation Risk:**
- [ ] Focused on essential information
- [ ] Avoiding unnecessary verbosity
- [ ] Right level of detail for audience

**Authority Bias:**
- [ ] Not assuming existing knowledge without verification
- [ ] Explaining concepts, not just stating them
- [ ] Providing context, not just instructions

---

## Step 5: Quality & Usability

**Clarity:**
- [ ] Language appropriate for audience
- [ ] Technical terms explained
- [ ] No ambiguous statements

**Accuracy:**
- [ ] Code examples will be tested
- [ ] Commands will be verified
- [ ] Paths and references will be checked

**Completeness:**
- [ ] All steps covered
- [ ] No "left as exercise" without clear reason
- [ ] Next steps or related docs referenced

**Maintainability:**
- [ ] Version-specific notes included
- [ ] Date/version information
- [ ] Easy to update structure

---

## Step 6: Synthesis & Plan

**Documentation Outline:**
$DOC_OUTLINE

**Quality Checks to Perform:**
$QUALITY_CHECKS

**Verification Needed:**
$VERIFICATION_NEEDED

**Revised Confidence (1-10):** $REVISED_CONFIDENCE

**Proceed with Documentation?** (YES/NO/CONDITIONAL)

---

## Output Format

```
## Critical Think: Before Documentation

### Step 1: Core Thesis
**What to Convey:** [Thesis]
**Key Information:** [List]
**Initial Confidence:** [X/10]

### Step 2: Assumptions
1. [Assumption] - [Verification method]
2. [Assumption] - [Verification method]
3. [Assumption] - [Verification method]

### Step 3: Logical Structure
**Structure:** [Outline with sections]
**Completeness:** [Check results]
**Issues:** [None or list]

### Step 4: AI Pitfalls
**Hallucination Risk:** [Check + verification plan]
**Happy Path Bias:** [Check + error scenarios]
**Over-Documentation:** [Check]
**Authority Bias:** [Check]

### Step 5: Quality & Usability
**Clarity:** [Assessment]
**Accuracy:** [Verification plan]
**Completeness:** [Check results]
**Maintainability:** [Plan]

### Step 6: Synthesis
**Outline:** [Detailed outline]
**Quality Checks:** [List of checks]
**Verification:** [What needs verification]
**Revised Confidence:** [X/10]
**Proceed:** [YES/NO/CONDITIONAL]
```

---

## After Documentation Generation

After generating documentation, run the **After Action** validation:

1. **Verify all code examples** (they should run without errors)
2. **Check all links and references**
3. **Verify technical accuracy**
4. **Test the documentation** (can someone follow it successfully?)
5. **Review for completeness** (did you miss anything?)
6. **Assess clarity** (will the target audience understand?)
