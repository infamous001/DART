import os
import json
import re
import sys
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI  
from text_extraction import text_extract
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reflexion_agent')))
from reflexion import run_reflexion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reinvoke')))
from reinvoke import run_reinvoke

llm = ChatOpenAI(model="gpt-4o", temperature=0)
topics_learnt=[]
issue_topics=[]
topics_cant_learnt=[]
total_questions=0
total_correct_questions=0
trajectory={}


############################################################################################################################################
def get_global_insights(file_path):
    with open(file_path, "r") as f:
        g_insights = json.load(f)

    global_insights = []
    for key, insights in g_insights.items():
        for _, insight in insights.items():
            global_insights.append(insight)

    return "\n".join(global_insights)

def summarise_local_insights(l_insights):
    prompt = f"""Please summarize the following insights. 
- Do not skip any important detail.
- Eliminate redundancy and overlap in the points.
- Present the summary in a clear and in a point-wise manner.

Insights:
{l_insights}
"""
    messages = [
        SystemMessage(content="You are a helpful and concise summarizer."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages).content.strip()
    return response

def get_local_insights(file_path):

    with open(file_path, "r") as f:
        l_insights = json.load(f)

    l_insights=l_insights["reflections"]

    return summarise_local_insights(l_insights)


###############################################################################################################

def PassageThought(student_query,global_insights,material,previous_topics):
  prompt = f"""
You are a thoughtful and experienced teacher helping a student understand a complex topic. Your goal is to design the next section of a lecture that builds on the student's prior learning, aligns with effective teaching principles, and is grounded in relevant material.

### Student Query
{student_query}

Over the years, you’ve developed general strategies and principles for effective teaching. These are captured as **global insights**, which you should follow carefully while planning your lecture.

### Global Insights
{global_insights}

You will now plan the **next section** of the lecture — just as a teacher would prepare a lesson in blocks. This means:
- Think carefully and decide **what the next section should cover**.
- Ensure the new section is **not a repetition** of anything listed in **Topics Already Covered**.
- Design the **flow, order, and structure** of this section.
- Use the **provided material** to support the design of this section — it will eventually be used to generate the lecture content itself.

### Material (reference)
{material}

### Topics Already Covered
{previous_topics}

### IMPORTANT
Respond **only** in JSON format with the following structure:

{{
  "title": "TITLE OF THE NEXT SECTION",
  "plan": "[ DETAILED PLAN FOR THIS SECTION — flow, structure, key points, etc ]"
}}
"""

  messages = [
        SystemMessage(content="You are a skilled teacher planning the next section of a lecture block-by-block."),
        HumanMessage(content=f"{prompt}")
    ]
  response= llm.invoke(messages).content.strip()
  try:
        parsed = json.loads(response)
        return parsed
  except json.JSONDecodeError:
        # Attempt to extract JSON using regex (fallback for malformed output)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to parse response as JSON:\n" + response)

def CorrectionPassageThought(student_query,global_insights,local_insights,material):
  prompt = f"""
You are a thoughtful and experienced teacher helping a student understand a complex topic they have previously struggled with. Your task is to design a lecture section that directly addresses the student's challenges, applies effective teaching strategies, and leverages the provided learning material.

### Student Query
{student_query}

### Topics of Difficulty
{issue_topics}

Over the years, you’ve developed a set of general teaching strategies based on experience and best practices. These are captured as **Global Insights** and should be followed closely while planning the lecture.

### Global Insights
{global_insights}

In addition, you have access to **Local Insights** — personalized observations and reflections about this student’s learning behavior. These insights highlight areas where the student is likely to struggle or succeed while learning the issue topics. Use them to anticipate potential misunderstandings and guide your teaching accordingly, addressing both weaknesses and strengths.

### Local Insights
{local_insights}

Now, plan the **next section** of the lecture as you would when preparing a structured, student-centered lesson. Make sure to:
- Decide **what the next section should cover**, ensuring all of the **issue topics** are included and only those topics are addressed.
- Carefully design the **structure and flow** of this section, including concept progression, explanations, and transitions.
- Incorporate the **reference material** to support the sequencing and depth of explanation.

### Reference Material
{material}

### IMPORTANT
Respond **only** in JSON format using the following structure:

{{
  "title": "TITLE OF THE NEXT SECTION",
  "plan": "[ DETAILED PLAN FOR THIS SECTION — flow, structure, key points, transitions, and teaching approach ]"
}}
"""

  messages = [
        SystemMessage(content="You are a skilled teacher planning the lecture."),
        HumanMessage(content=f"{prompt}")
    ]
  response= llm.invoke(messages).content.strip()
  try:
        parsed = json.loads(response)
        return parsed
  except json.JSONDecodeError:
        # Attempt to extract JSON using regex (fallback for malformed output)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to parse response as JSON:\n" + response)

################################################################################################################################################################################################################
def PassageCreation(material,plan,title):
  prompt=f"""
You are an expert teacher creating a comprehensive educational section on the following **topic**. Your goal is to explain the topic clearly, engagingly, and in a structured manner, using the **plan** as the guide for flow and organization. Make sure the tone is friendly, informative, and accessible to students who may be new to the subject.

Use the provided **material** as the factual and conceptual foundation for your explanation. Follow the **plan** closely.

### Inputs:
**Topic:**
{title}

**Reference Material:**
{material}

**Structured Plan:**
{plan}

Respond in the following JSON format:
{{
  "title": "{title}",
  "explanation": [
    {{
      "title": "Title of subsection",
      "explanation": "Detailed explanation of the corresponding subsection(one paragraph explaining the following topic in detail)"
    }},
    ...
  ]
}}

"""

  messages = [
        SystemMessage(content="You are a skilled teacher trying to create next section of a lecture"),
        HumanMessage(content=f"{prompt}")
    ]
  response=llm.invoke(messages).content.strip()
  try:
        parsed = json.loads(response)
        return parsed
  except json.JSONDecodeError:
        # Attempt to extract JSON using regex (fallback for malformed output)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to parse response as JSON:\n" + response)


def CorrectionPassageCreation(material,plan):
    prompt="""
You are an expert teacher creating a comprehensive educational section on the following **topic**. Your goal is to explain the topic clearly, engagingly, and in a structured manner, using the **plan** as the guide for flow and organization. Make sure the tone is friendly, informative, and accessible to students who may be new to the subject.

Use the provided **material** as the factual and conceptual foundation for your explanation. Follow the **plan** closely.

### Inputs:

**Topic:**
{title}

**Reference Material:**
{material}

**Structured Plan:**
{plan}


Respond in the following JSON format:
{{
  "title": "{title}",
  "explanation": "Detailed explanation of the corresponding topic(one paragraph explaining the following topic in detail)"
}}
""" 
    revision_explanations=[]
    for topic in issue_topics:
        messages = [
        SystemMessage(content="You are a skilled teacher trying to create the lecture for the asked topic"),
        HumanMessage(content=prompt.format(title=topic,material=material,plan=plan))]
        response=llm.invoke(messages).content.strip()
        try:
            parsed = json.loads(response)
            revision_explanations.append(parsed)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    temp=json.loads(json_match.group())
                    revision_explanations.append(temp)
                    continue
                except json.JSONDecodeError:
                    pass
            raise ValueError("Failed to parse response as JSON:\n" + response)
    return revision_explanations

###############################################################################################################################################################
def QuestionAgent(explanations):
    prompt = """
You are a teacher evaluating a student's understanding of recently taught **material**.

Your task is to generate **one multiple-choice question (MCQ)** at a time based on the content provided.

### Guidelines:
- The question should be meaningfully derived from the material, requiring actual understanding — not simple keyword matching or copy-pasting.
- Ensure the question tests **conceptual clarity**, not memorization.
- Provide exactly **4 answer options**.
- Include **only one correct answer**.
- Avoid overly complex language or ambiguity.

### Material:
{material}

### Output Format:
{{
    "question": "Your generated question goes here",
    "options": ["Option A", "Option B", "Option C", "Option D"]
}}
"""
    q_list = []
    for a in explanations:
        material = a["explanation"]
        title=a["title"]
        messages = [
            SystemMessage(content="You are a skilled teacher trying to test the student capablity by asking questions based on the taught material."),
            HumanMessage(content=prompt.format(material=material))
        ]
        response = llm.invoke(messages).content.strip()

        try:
            parsed = json.loads(response)
            parsed["material"]=material
            parsed["title"]=title
            q_list.append(parsed)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    q_list.append(parsed)
                except json.JSONDecodeError:
                    raise ValueError("Extracted JSON is still malformed:\n" + json_match.group())
            else:
                raise ValueError("Failed to parse response as JSON:\n" + response)

    return q_list

###############################################################################################################################################################
def StudentAgent(questions_list):
    print("Topic:", topics_learnt[-1])
    print()
    for idx, question in enumerate(questions_list, 1):
        print("Sub-Topic:",question["title"])
        topics_learnt.append(question["title"])
        print()
        print("Material:",question["material"])
        print()
        print(f"Q{idx}: {question['question']}\n")
        print("Options:", question["options"])
        print()
        answer = input("Your answer: ")
        question["answer"] = answer
        print()  # Blank line for readability

    return questions_list

def CorrectionStudentAgent(questions_list):
    print("Revision Time")
    print()
    print("Revision Topics: ",issue_topics)
    for idx, question in enumerate(questions_list, 1):
        print("Topic:",question["title"])
        print()
        print("Material:",question["material"])
        print()
        print(f"Q{idx}: {question['question']}\n")
        print("Options:", question["options"])
        print()
        answer = input("Your answer: ")
        question["answer"] = answer
        print()  # Blank line for readability

    return questions_list

###############################################################################################################################################################
def EvalAgent(qa_list,section_trajectory):
    global issue_topics
    global total_questions
    global total_correct_questions
    global topics_learnt
    score = 0
    prompt = """
You will be given a multiple-choice question along with the available options and a student's selected answer.

Your task is to evaluate whether the student's response is correct based on the context and explanation provided.

**Material**
{material}

**Question**
{question}

**Options**
{options}

**Student's Answer**
{answer}

Respond with **1** if the answer is correct, or **0** if the answer is incorrect. Respond with only the number — no explanation or extra text.
"""
    for qa in qa_list:

        sub_section={}
        messages = [
            SystemMessage(content="You are an evaluator. You must score the student's response using only 1 or 0."),
            HumanMessage(content=prompt.format(
                material=qa['material'],
                question=qa['question'],
                options=qa['options'],
                answer=qa['answer']
            ))
        ]
        sub_section['material']=qa['material']
        sub_section['question']=qa['question']
        sub_section['options']=qa['options']
        sub_section['student_answer']=qa['answer']

        response = llm.invoke(messages).content.strip()

        try:
            if(int(response)==0):
                issue_topics.append(qa['title'])

            if(int(response)==0):
                sub_section["result"]="incorrect"
            elif(int(response)==1):
                sub_section["result"]="correct"
                total_correct_questions=total_correct_questions+1

            score += int(response)
            total_questions=total_questions+1
        except ValueError:
            print(f"Invalid response from LLM: {response}")
        
        section_trajectory[qa["title"]]=sub_section

    final= score/len(qa_list)
    if(final>=0.7):
        issue_topics=[]
    else:
        topics_learnt = [x for x in topics_learnt if x not in issue_topics]
    return final

def CorrectionEvalAgent(qa_list,revision_trajectory):
    global issue_topics
    global topics_cant_learnt
    global topics_learnt
    score = 0
    prompt = """
You will be given a multiple-choice question along with the available options and a student's selected answer.

Your task is to evaluate whether the student's response is correct based on the context and explanation provided.

**Material**
{material}

**Question**
{question}

**Options**
{options}

**Student's Answer**
{answer}

Respond with **1** if the answer is correct, or **0** if the answer is incorrect. Respond with only the number — no explanation or extra text.
"""
    for qa in qa_list:
        sub_section={}
        messages = [
            SystemMessage(content="You are an evaluator. You must score the student's response using only 1 or 0."),
            HumanMessage(content=prompt.format(
                material=qa['material'],
                question=qa['question'],
                options=qa['options'],
                answer=qa['answer']
            ))
        ]
        sub_section['material']=qa['material']
        sub_section['question']=qa['question']
        sub_section['options']=qa['options']
        sub_section['student_answer']=qa['answer']

        response = llm.invoke(messages).content.strip()

        try:
            if(int(response)==1):
                topics_learnt.append(qa['title'])
            elif(int(response)==0):
                topics_cant_learnt.append(qa['title'])

            if(int(response)==0):
                sub_section["result"]="incorrect"
            elif(int(response)==1):
                sub_section["result"]="correct"
            
            revision_trajectory[qa["title"]]=sub_section

            score += int(response)
        except ValueError:
            print(f"Invalid response from LLM: {response}")

    final= score/len(qa_list)
    issue_topics=[]
    return final
############################################################################################################################################
def Endagent(material,topics_done):
    prompt = f"""You are given a **learning material** and a list of **topics already covered in class**.

Your task is to determine whether the covered topics fully account for the content in the material.

**Material:**
{material}

**Topics Covered:**
{topics_done}

If you believe the topics covered are sufficient to fully understand the material, respond with only:
**END**

If any relevant topic appears to be missing or only partially covered, respond with:
**NOT**
"""
    messages = [
            SystemMessage(content="You are a coverage checker. Decide if the given topics fully cover the material."),
            HumanMessage(content=prompt)
        ]
    response=llm.invoke(messages).content.strip()
    return response
###################################################################
def main():
    file_path = os.path.join(os.getcwd(), "expel_agent", "final_global_insights.json")
    global_insights=get_global_insights(file_path)
    student_query = input("Enter your query: ")
    notes_path=run_reinvoke(student_query)
    print(notes_path[0])
    material=text_extract(notes_path[0])

    check="NOT"
    while check!="END":
        section_trajectory={}
        a=PassageThought(student_query,global_insights,material,topics_learnt)
        section_trajectory["plan"]=a["plan"]
        topics_learnt.append(a["title"])
        b=PassageCreation(material,a["plan"],a["title"])
        c=QuestionAgent(b["explanation"])
        d=StudentAgent(c)
        score=EvalAgent(d,section_trajectory)
        print(score)
        if(score<0.7):
            revision_trajectory={}
            top=issue_topics
            run_reflexion(material,topics_learnt,issue_topics)
            file_path=os.path.join(os.getcwd(), "reflexion_agent", "local_insights.json")
            local_insights=get_local_insights(file_path)
            aa=CorrectionPassageThought(student_query,global_insights,local_insights,material)
            bb=CorrectionPassageCreation(material,aa["plan"])
            cc=QuestionAgent(bb)
            dd=CorrectionStudentAgent(cc)
            score=CorrectionEvalAgent(dd,revision_trajectory)
            section_trajectory["Revision"]=revision_trajectory
            print(score)
        trajectory[a["title"]]=section_trajectory
        topics_done=topics_learnt+topics_cant_learnt
        check=Endagent(material,topics_done)
        if "END" in check:
            if(total_correct_questions/total_questions>=0.7):
                with open("/Users/rohitjindal/Desktop/narratives-epfl/expel_agent/success.json", "r") as f:
                    data = json.load(f)
                data[notes_path[0]]=trajectory
                with open("/Users/rohitjindal/Desktop/narratives-epfl/expel_agent/success.json", "w") as f:
                    json.dump(data, f, indent=4)
            else:
                with open("/Users/rohitjindal/Desktop/narratives-epfl/expel_agent/fail.json", "r") as f:
                    data = json.load(f)
                data[notes_path[0]]=trajectory
                with open("/Users/rohitjindal/Desktop/narratives-epfl/expel_agent/fail.json", "w") as f:
                    json.dump(data, f, indent=4)
            print("DONE...")
            break

if __name__ == "__main__":
    main()


