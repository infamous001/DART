
import json
from openai import OpenAI
import re


client = OpenAI()  # set once globally

def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert insight extractor agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

with open('/Users/rohitjindal/Desktop/narratives-epfl/expel_agent/success.json') as f:
    success_data = json.load(f)

with open('/Users/rohitjindal/Desktop/narratives-epfl/expel_agent/fail.json') as f:
    fail_data = json.load(f)



C_compare = {}
for task_name1, trajectories1 in success_data.items():
    temp=[]
    for task_name2, trajectories2 in fail_data.items():
        if(task_name1==task_name2):
            temp.append([trajectories1,trajectories2])
        if(len(temp)>=3):
            break
    if(len(temp)!=0):
        C_compare[task_name1]=temp
    if(len(C_compare)>=3):
        break

C_success = {}
for task_name, trajectories in success_data.items():
    C_success[task_name1]=trajectories
    if(len(C_success)==5):
        break

insights_compare = {}
insights_success={}

def apply_operation(op_line,agent):
    match = re.match(r"(ADD|EDIT|UPVOTE|DOWNVOTE) (\d+): (.+)", op_line.strip())
    if not match:
        return
    if(agent=="compare"):
        op, rule_id, rule = match.groups()
        if op == "ADD":
            insights_compare[rule_id] = rule
        elif op == "EDIT" and rule_id in insights_compare:
            insights_compare[rule_id] = rule
        elif op == "UPVOTE" and rule_id in insights_compare:
            insights_compare[rule_id] += " (↑)"
        elif op == "DOWNVOTE" and rule_id in insights_compare:
            insights_compare[rule_id] += " (↓)"

    elif(agent=="success"):
        op, rule_id, rule = match.groups()
        if op == "ADD":
            insights_success[rule_id] = rule
        elif op == "EDIT" and rule_id in insights_success:
            insights_success[rule_id] = rule
        elif op == "UPVOTE" and rule_id in insights_success:
            insights_success[rule_id] += " (↑)"
        elif op == "DOWNVOTE" and rule_id in insights_success:
            insights_success[rule_id] += " (↓)"


def LLM_insights_compare(success, fail, current_rules,task_name):
    prompt =f"""
You are an advanced reasoning agent tasked with refining teaching strategies by analyzing past instructional outcomes. Your role is to critique and enhance the rules used to guide the teacher's Thought of Action — the planning and reasoning that shapes how material is delivered to students.

You will be given two prior instructional trajectories centered around teaching the topic: **{task_name}**.

- One **failed** (multiple questions were answered incorrectly)
- One **succeeded** (all questions answered correctly)

Each trajectory includes the explanation presented, the questions asked, and the students' responses.

FAILED TRAJECTORY:
{fail}

SUCCEEDED TRAJECTORY:
{success}

CURRENT RULES:
{json.dumps(current_rules, indent=2)}

Your task is to **analyze both trajectories**, identify differences in the instructional approach and student comprehension, and generate GENERAL and HIGH-LEVEL improvements. Use these insights to **add, edit, upvote, or downvote** rules. Your suggestions should help the teacher develop more robust teaching plans — applicable not just to this topic, but to **any future topic** with similar patterns of misunderstanding or success.

Focus especially on improving the **Thought of Action** — the implicit assumptions, ordering, phrasing, and strategy used during instruction.

Follow the format below:

<OPERATION> <RULE NUMBER>: <RULE>

The available operations are:
- **UPVOTE**: if the existing rule is highly relevant and should be reinforced.
- **DOWNVOTE**: if the rule is redundant, irrelevant, or contradicts better rules.
- **EDIT**: if a rule needs to be broadened, made clearer, or reframed to be more generally applicable.
- **ADD**: if a new rule is needed to address a gap revealed by the failure trajectory.

Each operation must strictly follow this structure:
    UPVOTE <EXISTING RULE NUMBER>: <EXISTING RULE>
    DOWNVOTE <EXISTING RULE NUMBER>: <EXISTING RULE>
    EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
    ADD <NEW RULE NUMBER>: <NEW RULE>

**Important Constraints**:
- Do not mention specific trials or student answers in the rules.
- Rules must be GENERAL and apply broadly across topics and learners.
- Use clear and concise phrasing.
- A maximum of 4 operations is allowed.
- Each existing rule may only be modified by one operation.

Your output should propose the most meaningful changes to improve future instructional effectiveness based on comparing the failed and successful teaching trajectories.
"""
    response = call_llm(prompt)
    for line in response.split('\n'):
        if line.strip():
            apply_operation(line,"compare")

def LLM_insights_success_only(success, current_rules,task_name):
    prompt = f"""
You are an advanced reasoning agent tasked with refining teaching strategies by analyzing patterns across **multiple successful instructional trajectories**. Your role is to critique and enhance the rules that guide the teacher's *Thought of Action* — the planning and reasoning that shape how material is delivered to students.

You will be given several successful instructional trajectories centered around teaching the topic: **{task_name}**.

Each trajectory includes the explanation presented, the questions asked, and the students' responses.

SUCCEEDED TRAJECTORIES:
{success}

CURRENT RULES:
{json.dumps(current_rules, indent=2)}

Your task is to **analyze patterns across the successful trajectories** and extract HIGH-LEVEL, GENERAL insights into effective teaching practices. Identify what *consistently contributed* to student understanding, engagement, and conceptual clarity. Use this analysis to **add, edit, upvote, or downvote** rules. Your recommendations should help the teacher develop stronger and more transferable instructional strategies — applicable not only to this topic, but to **any future topic with similar pedagogical characteristics**.

Focus on improving the **Thought of Action** — that is, the teacher’s implicit strategy: how concepts are introduced, how questions are ordered, how feedback is handled, and how learning is structured.

Follow the format below:

<OPERATION> <RULE NUMBER>: <RULE>

The available operations are:
- **UPVOTE**: if the existing rule is highly relevant and should be reinforced.
- **DOWNVOTE**: if the rule is redundant, irrelevant, or contradicts better rules.
- **EDIT**: if a rule needs to be broadened, made clearer, or reframed to be more generally applicable.
- **ADD**: if a new rule is needed to address a gap revealed by patterns across the successful trajectories.

Each operation must strictly follow this structure:
    UPVOTE <EXISTING RULE NUMBER>: <EXISTING RULE>
    DOWNVOTE <EXISTING RULE NUMBER>: <EXISTING RULE>
    EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
    ADD <NEW RULE NUMBER>: <NEW RULE>

**Important Constraints**:
- Do not mention specific examples or student responses from the trajectories.
- Rules must be GENERAL and applicable across different topics and student levels.
- Use clear, direct, and pedagogically meaningful phrasing.
- Propose no more than 4 operations.
- Each existing rule may only be targeted by one operation.

Your output should prioritize the most insightful changes to help the teacher consistently succeed across a wide range of topics.
"""
    response = call_llm(prompt)
    for line in response.split('\n'):
        if line.strip():
            apply_operation(line,"success")


for task_name,trajectories in C_compare.items():
    for trajectory in trajectories:
        LLM_insights_compare(trajectory[0],trajectory[1],insights_compare,task_name)

for task_name,trajectories in C_success.items():
    LLM_insights_success_only(trajectories,insights_success,task_name)



# === Final Output ===
print("\nFinal Extracted Insights:")
for rule_id, rule in insights_success.items():
    print(f"{rule_id}: {rule}")

for rule_id, rule in insights_compare.items():
    print(f"{rule_id}: {rule}")

final_dict = {
    "comparison_insights": insights_compare,
    "success_insights": insights_success
}

# Save to a file named 'output.json'
with open("expel_agent/final_global_insights.json", "w") as f:
    json.dump(final_dict, f, indent=4)
