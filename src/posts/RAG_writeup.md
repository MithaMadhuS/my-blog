
![Figure 1: RAG MEME](/my-blog/assets/RAG_meme1.png)

Lately, I’ve been noticing an undeniable buzz around Retrieval-Augmented Generation (RAG) everywhere I turn. Maybe it’s because it’s become a core part of what I do for a living—or maybe the universe just really wants me to pay attention. LinkedIn posts, YouTube tutorials, blog articles—you name it, everyone seems to be talking about it. The recent surge makes sense, especially with models like Gemini and GPT pushing context windows beyond one million tokens. That’s like giving your AI an incredible memory—but with great memory comes great responsibility.

But here’s the catch — most discussions barely scratch the surface. Everyone loves to talk about retrieval and generation, the basics. Some brave souls mention indexing and reranking. But when it comes to the nitty-gritty — how embedding techniques really influence performance or how to optimize RAG systems so they actually work well in real-world applications, avoid hallucination, and tackle other common problems — the conversation goes quiet.

That got me curious. I wanted to go beyond the buzzwords to understand what truly makes RAG tick. So I started reading, experimenting, and learning.

Now, I’m excited to share what I’ve discovered in this blog series. We’ll journey from the basics to advanced RAG architectures — covering query translation, routing methods, and query construction techniques like text-to-SQL, text-to-graph (using Cypher), text-to-vector databases, and multi-representation indexing.

We’ll dive deep into RAG fusion techniques, Interleave retrieval with CoT, semantic routing, advanced indexing strategies, re-retrieval, re-generation techniques, and even feedback generation — or as I like to call it, self-RAG (because sometimes even the AI needs a little self-reflection 🙂).

If you’ve been wondering what makes RAG genuinely powerful or how to build state-of-the-art RAG solutions that don’t just sound impressive on paper but actually deliver—avoiding hallucination and other pitfalls along the way—this series is for you. Let’s move beyond the basics and unlock the full potential of Retrieval-Augmented Generation.

The entire series consists of 4 parts. This is the first part of that journey — where we’ll lay the foundation, break down the core components of RAG, and set the stage for everything that follows.
Buckle up — we’re about to dive deep into the real mechanics of modern RAG systems, beyond the hype and into the how.

### What is RAG?
In simple terms, Retrieval-Augmented Generation (RAG) is a method that makes large language models (LLMs) smarter by giving them access to extra information. It works by finding the most useful documents or data for a question and showing them to the LLM, helping it give better answers. This approach is great for chatbots and Q&A systems that need up-to-date facts or expert knowledge.

RAG consists of three main components:  
1.Indexing  
2.Retrieval  
3.Generation  

**Indexing:**
Indexing is the process of organizing a large collection of documents so they can be quickly searched. It can involve SQL databases, graph databases, or vector stores. In RAG, it means converting text into a searchable format. This might be a simple keyword list or advanced graph embeddings. Effective indexing ensures fast and accurate retrieval of relevant data.
**Retrieval:**
This step finds the most relevant documents based on a user’s question. It can use keyword matching or semantic search techniques. The better the retrieval mechanism, the more relevant the context for generation.
**Generation:**
Below is a Python code snippet demonstrating a basic Retrieval-Augmented Generation (RAG) setup using OpenAI and LangChain. After executing each code block, I highly recommend visiting the LangSmith to explore how each step of the pipeline is executed in detail.

```python
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import bs4
import os

# Initialize Settings
url = "https://github.blog/ai-and-ml/generative-ai/what-is-retrieval-augmented-generation-and-what-does-it-do-for-generative-ai/"
db_dir = "chroma_db"
os.makedirs(db_dir, exist_ok=True)

# Load Website Content
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer("div"))
)
docs = loader.load()

# Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed with OpenAI Embeddings
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory=db_dir
)

# Retrieve Relevant Context
retriever = vectorstore.as_retriever()

# Define a Function for RAG QA
def run_rag(question: str):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an AI assistant. Use the following context to answer the question at the end.
If the answer is not contained within the context, respond with
"I'm sorry, but I don't have enough information to answer that."

Context:

{context}

Question: {question}

Answer:"""

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return llm(prompt.format(context=context, question=question))

# Example Usage
question = "What is the difference between RAG and Fine-tuning?"
response = run_rag(question)
print(response)
 ``` 

Now we will go thorough interesting RAG landscapes, interesting paper and techniques which I learned on how to do better RAG. I had broken down into different sections as shown in the image. Starting from first

![Figure 2: RAG MEME](/my-blog/assets/RAG_architecture.png)

### Query Translation
Query translation is the first step in an advanced Retrieval-Augmented Generation (RAG) pipeline. It involves rewriting a user’s query to improve search results. Since RAG relies on semantic search to retrieve relevant documents, poorly phrased queries can lead to confusion and missed answers. Users often express the same intent in different ways, which can hinder accurate retrieval.

By expanding and normalizing the query—expressing it in multiple forms—query translation helps the system better capture user intent. This approach generates multiple similar embeddings for the query, increasing the chances of matching the right documents, even when the original wording is unclear or ambiguous.

![Figure 3: RAG MEME](/my-blog/assets/query_translation.png)

There are several effective strategies for query translation, each leveraging Large Language Models (LLMs):

**Query Rewrite:** Reframe the input query into multiple variations to cover different perspectives, Combine multiple queries or query embeddings to improve retrieval accuracy (e.g. Multi Query, RAG Fusion)

**Least-to-Most Approach:** Make the query less abstract, breaking it down into more specific terms. (E.g. Interleave retrieval with CoT)

**Step-Back Prompting:** Make the query more abstract, encouraging broader interpretations.

We will explore each of these approaches in detail.

### Multi Query
Multi-query is a technique that takes a single search question and rephrases it into multiple variations, each approaching the same concept from different perspectives. This approach significantly increases the chances of retrieving relevant documents by exploring various linguistic and semantic interpretations of the query.

![Figure 4: RAG MEME](/my-blog/assets/multi_query.png)

#### Why Multi-Query Works
Multi-query works because of the nature of semantic embeddings, where text meanings are represented in high-dimensional vector space. Slight differences in wording can cause a query to appear in a different region of this space. By creating multiple versions of the same query, the system "fans out" across different embedding spaces. This increases the retrieval likelihood because similar documents aligned with these alternative phrasings might otherwise be missed.

#### How It Works
**1. Query Reformulation:**
The original query is rephrased in several ways. For example, a query like:

"Impact of climate change on agriculture"

can be rephrased as:

"How does climate change affect farming?"

"Effects of global warming on crop yields."

"Agricultural consequences of climate change."

**2. Document Retrieval:**
Each reformulated query is used to retrieve a set of relevant documents.

**3. Union of Results:**
The results from all reformulations are combined and deduplicated to form a broader, more relevant document set.

**4. Final Answer Generation (RAG):**
A language model generates the final answer using the context of all unique retrieved documents.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Function: Generate Multiple Query Perspectives
def generate_query_perspectives(question: str) -> list:
    template = '''You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}'''
    
    prompt = ChatPromptTemplate.from_template(template)
    generate_queries = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    return generate_queries.invoke({"question": question}).split("\n")

# Function: Get Unique Union of Retrieved Documents
def get_unique_union(documents: list[list]) -> list:
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# Function: Retrieval Augmented Generation (RAG)
def perform_rag(question: str, retriever) -> str:
    queries = generate_query_perspectives(question)
    retrieved_docs = retriever.map(queries)
    unique_docs = get_unique_union(retrieved_docs)

    # RAG Prompt
    context = "\n".join([doc["text"] for doc in unique_docs])
    template = '''Answer the following question based on this context:

{context}

Question: {question}'''
    
    rag_prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    
    final_answer = (rag_prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return final_answer

# Example Usage
if __name__ == '__main__':
    question = "Why does everyone talk about RAG?"

    # Replace 'retriever' with your actual retriever instance
    retriever = RunnablePassthrough()  # Placeholder for the retriever

    answer = perform_rag(question, retriever)
    print("Final Answer:", answer)
 ``` 

### RAG Fusion
RAG Fusion is an enhanced version of the multi-query retrieval technique. While multi-query simply merges documents retrieved from different query variations, RAG Fusion adds an intelligent ranking step to prioritize the most relevant results.

![Figure 4: RAG MEME](/my-blog/assets/rag_fusion.png)

#### Why RAG Fusion Works
RAG Fusion works because it goes beyond basic retrieval by using a ranking mechanism that identifies the most relevant documents across multiple query variations. It does this using Reciprocal Rank Fusion (RRF), which assigns higher scores to documents that appear higher in the ranking lists during retrieval part of each query, ensuring that the most relevant content is prioritized.

#### Why RAG Fusion Works
RAG Fusion goes beyond basic document retrieval by identifying the most relevant documents across multiple query variations. Instead of treating all retrieved documents equally, it uses a method called Reciprocal Rank Fusion (RRF) to score and rank documents based on how often and how highly they appear across the different query results. This helps highlight the most reliable and consistent information.

#### How RAG Fusion Works
Step-1 : Generate Multiple Query Variations
Several variations of the original query are generated, such as through paraphrasing or rewording, to increase the diversity of document retrieval.

Step-2 : Retrieve Documents for Each Query
Each query is submitted to a retrieval system (e.g., search engine or vector database), which returns a ranked list of relevant documents.

Step-3 : Reciprocal Rank Fusion (RRF) Scoring
Each document is assigned a score based on its position (rank) in the list of results using the following formula:

**Score = 1 / (rank + k)**

**rank:** The position of the document in the results list (starting from 1).  
**k:** A smoothing constant (commonly set to 60) to prevent division by zero and ensure even lower-ranked documents receive a small score.
The higher a document ranks in a list, the higher the score it receives.

Step-4 : Aggregate Scores Across All Queries
If a document appears in multiple result lists, its scores are summed (or averaged). This boosts the total score of documents that are both highly ranked and appear across multiple queries.

Step-5 : Select Top-Ranked Documents
After scoring and aggregation, the documents are sorted by their final scores. The top-ranked documents are selected for the next step, such as being used in a Retrieval-Augmented Generation (RAG) system to generate an answer.

```python
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter

# Function to generate multiple search queries from a single question
def generate_queries(question: str, llm: ChatOpenAI, num_queries: int = 4) -> List[str]:
    prompt_text = (
        "You are a helpful assistant that generates multiple search queries based on a single input query.\n"
        f"Generate {num_queries} search queries related to: {question}\n"
        "Output:"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    response = llm.invoke(prompt.format_prompt(question=question).to_messages())
    parsed = StrOutputParser().parse(response)
    queries = [q.strip() for q in parsed.split("\n") if q.strip()]
    return queries

# Simple Reciprocal Rank Fusion function to combine ranked lists of documents
def reciprocal_rank_fusion(results: List[List[dict]], k: int = 60) -> List[dict]:
    scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = dumps(doc)
            scores[key] = scores.get(key, 0) + 1 / (rank + k)
    # Sort by fused scores (highest first)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [loads(doc_str) for doc_str, _ in sorted_docs]

# Combine texts of documents to form context for the final answer
def build_context(documents: List[dict], max_length: int = 4000) -> str:
    context = ""
    for doc in documents:
        text = doc.get("text") or doc.get("content") or ""
        if len(context) + len(text) > max_length:
            break
        context += text + "\n"
    return context.strip()

# Generate the final answer using the LLM based on the context and question
def generate_final_answer(question: str, context: str, llm: ChatOpenAI) -> str:
    prompt_text = (
        "Answer the following question based on this context:\n\n"
        "{context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    response = llm.invoke(prompt.format_prompt(question=question, context=context).to_messages())
    return StrOutputParser().parse(response)

# Main function to run the RAG-Fusion pipeline
def run_rag_fusion(question: str, retriever, llm: ChatOpenAI):
    # Step 1: Generate related queries
    queries = generate_queries(question, llm)

    # Step 2: Retrieve documents for each query
    retrieved_docs = retriever.map(queries)

    # Step 3: Fuse all retrieved documents into a single ranked list
    fused_docs = reciprocal_rank_fusion(retrieved_docs)

    # Step 4: Build context from fused documents
    context = build_context(fused_docs)

    # Step 5: Generate the final answer
    answer = generate_final_answer(question, context, llm)
    return answer

# Example usage
if __name__ == "__main__":
    question = "Why does everyone talk about RAG?"
    retriever = ...  # Your retriever instance here
    llm = ChatOpenAI(temperature=0)
    answer = run_rag_fusion(question, retriever, llm)
    print("Final Answer:\n", answer)
``` 

### Least to Most:
Least-to-Most prompting is a technique that helps language models tackle complex query by breaking them down into smaller, manageable steps. This method works in two main stages:

![Figure 4: RAG MEME](/my-blog/assets/least_most.png)

**1. Decomposition** - The prompt in this stage contains constant examples that demonstrate the decomposition, followed by the specific question to be decomposed.

**2. Subproblem solving** - The prompt in this stage consists of three parts: (1) constant examples demonstrating how subproblems are solved; (2) a potentially empty list of previously answered subquestions and generated solutions, and (3) the question to be answered next.

In the example shown in Figure, the language model is first asked to decompose the original problem into subproblems. The prompt that is passed to the model consists of examples that illustrate how to decompose complex problems (which are not shown in the figure), followed by the specific problem to be decomposed (as shown in the figure). The language model figures out that the original problem can be solved via solving an intermediate problem “How long does each trip take?”.
In the next phase, we ask the language model to sequentially solve the subproblems from the problem decomposition stage. The original problem is appended as the final subproblem. The solving starts from passing to the language model a prompt that consists of examples that illustrate how problems are solved (not shown in the figure), followed by the first subproblem “How long does each trip take?”. We then take the answer generated by the language model (“... each trip takes 5 minutes.”)
and construct the next prompt by appending the generated answer to the previous prompt, followed by the next subproblem, which happens to be the original problem in this example. The new prompt is then passed back to the language model, which returns the final answer.

we can achieve this using both recursive and parallel answering approach

#### Recursive Answering Approach:
In this approach, questions are processed sequentially, with each question being answered in the context of the previous Q&A pair and any newly fetched information. This method ensures continuity, maintaining a coherent perspective across all responses. It is particularly effective for complex queries where maintaining a consistent narrative is crucial and questions are dependent on each other

![Figure 4: RAG MEME](/my-blog/assets/recursive_app.png)

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Function: Generate Multiple Query Perspectives
def generate_query_perspectives(question: str) -> list:
    template = '''You are an AI language model designed to help improve document retrieval. Your task is to create five rephrased versions of the given user question, each aiming to capture a different perspective. These variations will help overcome the limitations of distance-based similarity search in a vector database. Provide these alternative questions separated by newlines. Original question: {question}'''
    prompt = ChatPromptTemplate.from_template(template)
    generate_queries = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    return generate_queries.invoke({"question": question}).split("\n")

# Function: Get Unique Union of Retrieved Documents
def get_unique_union(documents: list[list]) -> list:
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# Function: Retrieval Augmented Generation (RAG) with Decomposition
def perform_rag_decomposition(question: str, retriever) -> str:
    # Generate sub-questions (decomposition)
    sub_q_template = '''You are a helpful assistant that generates multiple sub-questions related to an input question.\nGenerate multiple sub-questions related to: {question}'''
    sub_q_prompt = ChatPromptTemplate.from_template(sub_q_template)
    sub_questions = (sub_q_prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"question": question}).split("\n")

    # Collect Q-A pairs
    q_a_pairs = ""
    for sub_q in sub_questions:
        retrieved_docs = retriever.map([sub_q])
        unique_docs = get_unique_union(retrieved_docs)
        context = "\n".join([doc["text"] for doc in unique_docs])
        template = '''Answer the following question based on this context:\n\n{context}\n\nQuestion: {sub_q}'''
        answer = (ChatPromptTemplate.from_template(template) | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"context": context, "sub_q": sub_q})
        q_a_pairs += f"Question: {sub_q}\nAnswer: {answer}\n---\n"

    # Final answer with Q-A pairs
    final_prompt = '''Based on the following question-answer pairs, answer the main question:\n\n{q_a_pairs}\n\nMain Question: {question}'''
    final_answer = (ChatPromptTemplate.from_template(final_prompt) | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"q_a_pairs": q_a_pairs, "question": question})
    return final_answer

# Example Usage
if __name__ == '__main__':
    question = "How RAG retrieves data from Vector databases"
    retriever = RunnablePassthrough()  # Placeholder for the retriever
    answer = perform_rag_decomposition(question, retriever)
    print("Final Answer:", answer)
```

#### Parallel Answering Approach:
This approach involves breaking down a user query into multiple nuanced sub-questions and addressing them independently in parallel. The individual answers are then synthesized to create a comprehensive and contextually rich response. Given the quality of the sub-questions, this method is highly efficient for most scenarios, delivering fast and accurate results and widely used when questions/sub queries are independent of each other

![Figure 4: RAG MEME](/my-blog/assets/parallel_app.png)

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Function: Generate Multiple Query Perspectives
def generate_query_perspectives(question: str) -> list:
    template = '''You are an AI language model designed to help improve document retrieval. Your task is to create five rephrased versions of the given user question, each aiming to capture a different perspective. These variations will help overcome the limitations of distance-based similarity search in a vector database. Provide these alternative questions separated by newlines. Original question: {question}'''
    prompt = ChatPromptTemplate.from_template(template)
    generate_queries = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    return generate_queries.invoke({"question": question}).split("\n")

# Function: Get Unique Union of Retrieved Documents
def get_unique_union(documents: list[list]) -> list:
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# Function: Retrieve and Perform RAG with Decomposition
def retrieve_and_rag(question, retriever, prompt_rag) -> tuple:
    sub_q_template = '''You are a helpful assistant that generates multiple sub-questions related to an input question.\nGenerate multiple sub-questions related to: {question}'''
    sub_q_prompt = ChatPromptTemplate.from_template(sub_q_template)
    sub_questions = (sub_q_prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"question": question}).split("\n")

    answers = []
    for sub_q in sub_questions:
        retrieved_docs = retriever.get_relevant_documents(sub_q)
        context = "\n".join([doc["text"] for doc in retrieved_docs])
        answer = (prompt_rag | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"context": context, "question": sub_q})
        answers.append(answer)

    return answers, sub_questions

# Function: Format Q&A Pairs
def format_qa_pairs(questions, answers):
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

# Example Usage
if __name__ == '__main__':
    question = "How RAG retrieves data from vector databases ?"
    retriever = RunnablePassthrough()  # Placeholder for the retriever

    # Load RAG prompt from hub (or define it directly)
    prompt_rag = ChatPromptTemplate.from_template("Answer the following question using this context: \n\n{context}\n\nQuestion: {question}")

    answers, sub_questions = retrieve_and_rag(question, retriever, prompt_rag)
    context = format_qa_pairs(sub_questions, answers)

    final_prompt = ChatPromptTemplate.from_template("Here is a set of Q+A pairs:\n\n{context}\n\nUse these to synthesize an answer to the question: {question}")
    final_answer = (final_prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"context": context, "question": question})

    print("Final Answer:\n", final_answer)
```

### Step Back
The Step-Back Approach adopts a contrasting strategy to what we have studied so far, focusing on asking abstract questions. Step-Back Prompting is based on the observation that many tasks involve numerous details, making it challenging for language models to access relevant information effectively. A step-back question is an abstracted version of the original question, providing a broader perspective. For example, rather than directly asking, "Which school did Estella Leopold attend during a specific period?", a step-back question would inquire about her "education history," a higher-level concept that encompasses the original query. By first addressing this broader question, the model can gather all necessary information to accurately determine which school she attended during a particular period. This approach simplifies the reasoning process, reducing the chances of errors in the intermediate steps typically seen in Chain-of-Thought reasoning.

![Figure 4: RAG MEME](/my-blog/assets/step_back.png)

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Function: Generate Multiple Query Perspectives
def generate_query_perspectives(question: str) -> list:
    template = '''You are an AI language model designed to help improve document retrieval. Your task is to create five rephrased versions of the given user question, each aiming to capture a different perspective. These variations will help overcome the limitations of distance-based similarity search in a vector database. Provide these alternative questions separated by newlines. Original question: {question}'''
    prompt = ChatPromptTemplate.from_template(template)
    generate_queries = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    return generate_queries.invoke({"question": question}).split("\n")

# Function: Get Unique Union of Retrieved Documents
def get_unique_union(documents: list[list]) -> list:
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# Function: Retrieve and Perform RAG with Decomposition
def retrieve_and_rag(question, retriever, prompt_rag) -> tuple:
    sub_q_template = '''You are a helpful assistant that generates multiple sub-questions related to an input question.\nGenerate multiple sub-questions related to: {question}'''
    sub_q_prompt = ChatPromptTemplate.from_template(sub_q_template)
    sub_questions = (sub_q_prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"question": question}).split("\n")

    answers = []
    for sub_q in sub_questions:
        retrieved_docs = retriever.get_relevant_documents(sub_q)
        context = "\n".join([doc["text"] for doc in retrieved_docs])
        answer = (prompt_rag | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"context": context, "question": sub_q})
        answers.append(answer)

    return answers, sub_questions

# Function: Generate Step-Back Questions with Few-Shot Examples
def generate_step_back_questions(question: str) -> str:
    examples = [
        {"input": "Who won the Nobel Prize in Physics in 2021?", "output": "What are the notable achievements of Nobel Prize winners in Physics?"},
        {"input": "How does the Mars rover Perseverance collect rock samples?", "output": "How do Mars rovers collect and analyze rock samples?"}
    ]

    few_shot_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"),
        *[("human", ex["input"]) if i % 2 == 0 else ("ai", ex["output"]) for i, ex in enumerate(sum([[e] * 2 for e in examples], []))],
        ("user", "{question}")
    ])

    return (few_shot_prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"question": question})

# Function: Format Q&A Pairs
def format_qa_pairs(questions, answers):
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

# Example Usage
if __name__ == '__main__':
    question = "How RAG retrieves information from vector database?"
    retriever = RunnablePassthrough()  # Placeholder for the retriever

    # Load RAG prompt from hub (or define it directly)
    prompt_rag = ChatPromptTemplate.from_template("Answer the following question using this context: \n\n{context}\n\nQuestion: {question}")
    answers, sub_questions = retrieve_and_rag(question, retriever, prompt_rag)
    context = format_qa_pairs(sub_questions, answers)

    # Generate step-back question
    step_back_question = generate_step_back_questions(question)
    print("Step-Back Question:\n", step_back_question)

    # Final synthesis
    final_prompt = ChatPromptTemplate.from_template("Here is a set of Q+A pairs:\n\n{context}\n\nUse these to synthesize an answer to the question: {question}")
    final_answer = (final_prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"context": context, "question": question})

    print("Final Answer:\n", final_answer)
```

### Conclusion
In this first part of our Beyond Basics RAG series, we explored the foundational layers of a robust Retrieval-Augmented Generation system. We went beyond the standard definition of RAG to uncover how query translation techniques—like multi-query prompting, RAG fusion, step-back prompting, and the least-to-most approach—can dramatically influence the quality and accuracy of generated outputs. These methods aren’t just clever tricks; they play a critical role in shaping how your system interprets intent, retrieves relevant information, and ultimately delivers grounded, useful responses.

Understanding these techniques is essential, because they form the connective tissue between user input and effective retrieval. Without thoughtful query construction, even the most advanced retrieval engine can fall short. These strategies help mitigate hallucination, ensure better alignment with user intent, and create more context-aware generation—all key to building a truly performant RAG system.

But this is just the beginning.

In the next parts of the series, we’ll dive into what many consider the most exciting (and often overlooked) dimensions of RAG systems: advanced indexing strategies, semantic and hybrid retrieval, re-ranking, re-retrieval, and feedback mechanisms like self-RAG. If you're looking to build RAG solutions that are not just functional but truly state-of-the-art, you won’t want to miss what’s coming next.

This concludes part one of our series on RAG beyond the basics.

Stay tuned—things are about to get really interesting.

![Figure 4: RAG MEME](/my-blog/assets/final_meme.png)

### Resource
1. **RAG from Scratch** by Langchain [YouTube](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
2. **Least-to-Most** by Google [arxiv](https://arxiv.org/pdf/2205.10625)
3. **Interleaving Retrieval** by Stony Brook [arxiv](https://arxiv.org/pdf/2205.106254)
4. Lance Martin (LangChain), **Learn RAG From Scratch** [Github](https://github.com/langchain-ai/rag-from-scratch)