from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import random
import json
from text_extraction import text_extract
from typing import TypedDict, List, Optional


class Student(TypedDict):
    age:int
    gender:str
    race:str
    education_level:str
    education_subject:str
    occupation:str
    topics_learnt:List[str]
    issue_topic:str

class ReflexionState(TypedDict):
    task_description: str
    material: str
    explanation: Optional[str] 
    short_term_trajectory: List[str]  
    attempt: Optional[str]           
    student_answers: List[str]       
    eval_result: Optional[str]       
    reflections: List[str]           
    passed: bool                     
    trial: int  
    student:Student                     

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def actor_node(state: ReflexionState) -> ReflexionState:
    reflection_str = "\n".join(state["reflections"])
    short_term_str = "\n".join(state["short_term_trajectory"])
    if state["explanation"] is None:
        expl_prompt = (
        f""" 
        You are a supportive and skilled teacher. Your goal is to help a student clearly understand the topic(s) they are struggling with by using the provided **Material**. 
        Please carefully read the **Material** and generate a clear, concise, and engaging explanation in **a few well-structured paragraphs**, focusing only on the specific **Issue Topic(s)** mentioned below.
        While generating the explanation make sure you incoporate the Reflections from previous iterations(if any)

        **Material**:
        {state['material']}

        **Topic(s) the student is struggling with**:
        {state['student']['issue_topic']}

        **Reflections from earlier trials (long-term memory)**:
        {reflection_str}
        """

        
        )
        explanation = llm.invoke(expl_prompt).content.strip()

        q_prompt = (f"""
Only using the **Explanation** below, create one **multiple-choice question** designed to test a student's understanding.
- Provide **exactly four answer options (A–D)**.
- The correct answer must be either explicitly stated in the **Explanation** or clearly inferable from it.
- Return **only** the question and the four answer choices — do not include the correct answer or any additional notes.

Explanation:
{state['explanation']}
"""
        )
        question = llm.invoke(q_prompt).content.strip()
        attempt = f"{question}"  
        return {
            **state,
            "explanation": explanation,
            "attempt": attempt,
        }

    q_prompt = f"""
Only using the explanation below, create one **multiple-choice question** with **exactly four options (A-D)** to assess the student's understanding.

Requirements:
- The question must be **different** from any in the **Recent Q&A pairs**.
- The correct answer must be **clearly stated or inferable** from the **Explanation**.
- Return **only** the question and the four answer choices — do not include the correct answer or any extra commentary.

Explanation:
{state["explanation"]}

Recent Q&A pairs:
{short_term_str}
"""

    response = llm.invoke(q_prompt)
    return {**state, "attempt": response.content}


def student_node(state: ReflexionState) -> ReflexionState:
    should_answer_correctly = random.random() > 0.7  #~30% chance of answering correctly

    base_prompt = f"""
You're a student trying to learn a challenging topic. Here's a snapshot of your background:

- Age: {state['student']['age']}
- Gender: {state['student']['gender']}
- Race: {state['student']['race']}
- Education: {state['student']['education_level']} in {state['student']['education_subject']}
- Occupation: {state['student']['occupation']}
- Understood: You know have command on the following topics: {state['student']['topics_learnt']}
- Current Struggle: You're having difficulty understanding following topics: {state['student']['issue_topic']}

As a student, you behave like a normal human learner — you're curious, but not perfect. You can sometimes:
- Be overconfident and give a wrong answer.
- Get confused or misinterpret explanations.
- Take a wild guess if you're unsure.
- Try to reason through a question, but make mistakes along the way.

These human-like tendencies mean you **might answer questions incorrectly** sometimes, just like any real student who’s still learning.

---

**Explanation Given:**  
{state['explanation']}

**Question to Answer:**  
{state['attempt']}
"""


    if should_answer_correctly:
        prompt = base_prompt + "\nTry your best to answer this question correctly, based on your understanding."
    else:
        prompt = base_prompt + "\nYou are confused or unsure. Give an answer that is incorrect, flawed, or based on a guess."

    response = llm.invoke(prompt)

    new_qa = f"""
    Question: {state['attempt']} 
    Student Answer: {response.content}"""
    
    return {
        **state,
        "student_answers": state["student_answers"] + [response.content],
        "short_term_trajectory": state["short_term_trajectory"] + [new_qa],
    }

MAX_TRIALS=1
def evaluator_node(state: ReflexionState) -> ReflexionState:
    if len(state["student_answers"]) < 5:
        return state

    if state["trial"] >= MAX_TRIALS:
        # print(f"⚠️ Reached maximum trials ({MAX_TRIALS}). Ending with FAIL.")
        return {**state, "eval_result": "FAIL", "passed": True}
    
    short_term_str = "\n".join(state["short_term_trajectory"])
    prompt = f"""
Evaluate whether the student correctly answered **all five questions** based on the explanation of the topic provided below.
Use only the explanation and the Q&A pairs to make your judgment. Respond strictly with **PASS** if all answers are correct, or **FAIL** otherwise — no extra text.

Explanation:
{state["explanation"]}

Student Q&A Pairs:
{short_term_str}
"""

    result = llm.invoke(prompt).content.strip()
    return {**state, "eval_result": result, "passed": result.startswith("PASS")}


def reflection_node(state: ReflexionState) -> ReflexionState:
    short_term_str = "\n".join(state["short_term_trajectory"])
    prompt = f"""
The student did not answer all 5 questions correctly based on your explanation of the material.

Material:
{state['material']}

Explanation you provided:
{state["explanation"]}

Student Q&A history:
{short_term_str}

Reflect on why the student may have struggled to understand the topic. Identify potential gaps or shortcomings in your explanation, and suggest how it could be improved in the next iteration to enhance clarity, engagement, or depth. Provide a thoughtful and constructive feedback.
"""
    feedback = llm.invoke(prompt).content
    return {
        **state,
        "explanation":None,
        "reflections": state["reflections"] + [feedback],
        "trial": state["trial"] + 1,
        "student_answers": [],
        "short_term_trajectory": [],
        "attempt": None,
    }

builder = StateGraph(ReflexionState)
builder.add_node("Actor", actor_node)
builder.add_node("Student", student_node)
builder.add_node("Evaluator", evaluator_node)
builder.add_node("Reflect", reflection_node)

builder.set_entry_point("Actor")
builder.add_edge("Actor", "Student")
builder.add_edge("Student", "Evaluator")
builder.add_conditional_edges(
    "Evaluator",
    lambda s: "END" if s["passed"] else ("Reflect" if len(s["student_answers"]) >= 5 else "Actor")
)
builder.add_edge("Reflect", "Actor")
builder.set_finish_point("Evaluator")

graph = builder.compile()

# === Run Example ===
# pdf_path = "/Users/rohitjindal/Desktop/narratives-epfl/reinvoke/notes/1|decisionTrees.pdf"
# extracted_text = text_extract(pdf_path)

# material="""Applications such as real-time graphics do not involve much branching compared to the computational workload that a CPU usually encounters. For example, each vertex in the same rigid object will be multiplied by the same matrix; there is no need to evaluate an if statement per vertex to determine which matrix to multiply by. The computations are also entirely independent of each other, and thus may be parallelized easily. The computations also involve processing massive buffers of memory, containing bitmaps describing the texture (color pattern) of each object to be rendered.

# Together, this results in graphics cards having been designed to have a high degree of parallelism and high memory bandwidth, at the cost of having a lower clock speed and less branching capability relative to traditional CPUs.

# Neural network algorithms require the same performance characteristics as the real-time graphics algorithms described above. Neural networks usually involve large and numerous buffers of parameters, activation values, and gradient values, each of which must be completely updated during every step of training. These buffers are large enough to fall outside the cache of a traditional desktop computer, so the memory bandwidth of the system often becomes the rate-limiting factor.

# GPUs offer a compelling advantage over CPUs because of their high memory bandwidth. Neural network training algorithms typically do not involve much branching or sophisticated control, so they are appropriate for GPU hardware. Since neural networks can be divided into multiple individual “neurons” that can be processed independently from the other neurons in the same layer, neural networks easily benefit from the parallelism of GPU computing.

# GPU hardware was originally so specialized that it could be used only for graphics tasks. Over time, GPU hardware became more flexible, allowing custom subroutines to be used to transform the coordinates of vertices or to assign colors to pixels. In principle, there was no requirement that these pixel values actually be based on a rendering task. These GPUs could be used for scientific computing by writing the output of a computation to a buffer of pixel values.

# Steinkrau et al. (2005) implemented a two-layer fully connected neural network on a GPU and reported a three-times speedup over their CPU-based baseline. Shortly thereafter, Chellapilla et al. (2006) demonstrated that the same technique could be used to accelerate supervised convolutional networks.

# The popularity of graphics cards for neural network training exploded after the advent of general purpose GPUs. These GP-GPUs could execute arbitrary code, not just rendering subroutines. NVIDIA’s CUDA programming language provided a way to write this arbitrary code in a C-like language. With their relatively convenient programming model, massive parallelism, and high memory bandwidth, GP-GPUs now offer an ideal platform for neural network programming.

# This platform was rapidly adopted by deep learning researchers soon after it became available (Raina et al., 2009; Ciresan et al., 2010).

# Writing efficient code for GP-GPUs remains a difficult task best left to specialists. The techniques required to obtain good performance on GPU are very different from those used on CPU. For example, good CPU-based code is usually designed to read information from the cache as much as possible. On GPU, most writable memory locations are not cached, so it can actually be faster to compute the same value twice, rather than compute it once and read it back from memory.

# GPU code is also inherently multithreaded and the different threads must be coordinated with each other carefully. For example, memory operations are faster if they can be coalesced. Coalesced reads or writes occur when several threads can each read or write a value that they need simultaneously, as part of a single memory transaction.

# Different models of GPUs are able to coalesce different kinds of read patterns and different kinds of write patterns. Typically, memory operations are easier to coalesce if among n threads, thread i accesses byte i + j of memory, and j is a multiple of some power of 2. The exact specifications differ between models of GPU.

# Another common consideration for GPUs is making sure that each thread in a group executes the same instruction simultaneously. This means that branching can be difficult on GPU. Threads are divided into small groups called warps. Each thread in a warp executes the same instruction during each cycle, so if different threads within the same warp need to execute different code paths, these different code paths must be traversed sequentially rather than in parallel.

# Because of the difficulty of writing high-performance GPU code, researchers should structure their workflow to avoid needing to write new GPU code to test new models or algorithms. Typically, one can do this by building a software library of high-performance operations like convolution and matrix multiplication, then specifying models in terms of calls to this library of operations.

# For example, the machine learning library Pylearn2 (Goodfellow et al., 2013c) specifies all its machine learning algorithms in terms of calls to Theano (Bergstra et al., 2010; Bastien et al., 2012) and cuda-convnet (Krizhevsky, 2010), which provide these high-performance operations. This factored approach can also ease support for multiple kinds of hardware. For example, the same Theano program can run on either CPU or GPU, without needing to change any of the calls to Theano itself.

# Other libraries like TensorFlow (Abadi et al., 2015) and Torch (Collobert et al., 2011b) provide similar features."""

def run_reflexion(material,topics_learnt,topics_issue):
 initial_state: ReflexionState = {
    "task_description": "Teach the student based on the provide material.",
    "material":  material,
    "explanation": None,
    "short_term_trajectory": [],
    "attempt": None,
    "student_answers": [],
    "eval_result": None,
    "reflections": [],
    "passed": False,
    "trial": 0,
    "student": {
        "age": 50,
        "gender": "Female",
        "race": "Black",
        "education_level": "High School",
        "education_subject": "English",
        "occupation": "Housewife",
        "topics_learnt": topics_learnt,
        "issue_topic": topics_issue
    }
 }

 final_state = graph.invoke(initial_state,config={"recursion_limit": 100})
#  print(f"\n✅ Evaluation: {final_state['eval_result']} in {final_state['trial'] + 1} trial(s)")
#  print(f"\n📘 Final Reflection(s):\n" + "\n".join(final_state['reflections']))

 with open("/Users/rohitjindal/Desktop/narratives-epfl/reflexion_agent/local_insights.json", "w") as f:
    json.dump(final_state, f, indent=4)


