from datasets import load_dataset
from semantic_router.encoders import HuggingFaceEncoder
from semantic_chunkers import StatisticalChunker

# 1. Load a document (or use your own text)
dataset = load_dataset("jamescalam/ai-arxiv2", split="train")
document = dataset[0]["content"][:5000]  # Use first 5000 characters for speed
document = """
Multi-agent collaboration is the last of the four key AI agentic design patterns that I’ve described in recent letters. Given a complex task like writing software, a multi-agent approach would break down the task into subtasks to be executed by different roles — such as a software engineer, product manager, designer, QA (quality assurance) engineer, and so on — and have different agents accomplish different subtasks.

Different agents might be built by prompting one LLM (or, if you prefer, multiple LLMs) to carry out different tasks. For example, to build a software engineer agent, we might prompt the LLM: “You are an expert in writing clear, efficient code. Write code to perform the task . . ..” 

It might seem counterintuitive that, although we are making multiple calls to the same LLM, we apply the programming abstraction of using multiple agents. I’d like to offer a few reasons:

It works! Many teams are getting good results with this method, and there’s nothing like results! Further, ablation studies (for example, in the AutoGen paper cited below) show that multiple agents give superior performance to a single agent. 
Even though some LLMs today can accept very long input contexts (for instance, Gemini 1.5 Pro accepts 1 million tokens), their ability to truly understand long, complex inputs is mixed. An agentic workflow in which the LLM is prompted to focus on one thing at a time can give better performance. By telling it when it should play software engineer, we can also specify what is important in that role’s subtask. For example, the prompt above emphasized clear, efficient code as opposed to, say, scalable and highly secure code. By decomposing the overall task into subtasks, we can optimize the subtasks better.
Perhaps most important, the multi-agent design pattern gives us, as developers, a framework for breaking down complex tasks into subtasks. When writing code to run on a single CPU, we often break our program up into different processes or threads. This is a useful abstraction that lets us decompose a task, like implementing a web browser, into subtasks that are easier to code. I find thinking through multi-agent roles to be a useful abstraction as well.
Proposed ChatDev architecture, illustrated.
In many companies, managers routinely decide what roles to hire, and then how to split complex projects — like writing a large piece of software or preparing a research report — into smaller tasks to assign to employees with different specialties. Using multiple agents is analogous. Each agent implements its own workflow, has its own memory (itself a rapidly evolving area in agentic technology: how can an agent remember enough of its past interactions to perform better on upcoming ones?), and may ask other agents for help. Agents can also engage in Planning and Tool Use. This results in a cacophony of LLM calls and message passing between agents that can result in very complex workflows. 

While managing people is hard, it's a sufficiently familiar idea that it gives us a mental framework for how to "hire" and assign tasks to our AI agents. Fortunately, the damage from mismanaging an AI agent is much lower than that from mismanaging humans! 

Emerging frameworks like AutoGen, Crew AI, and LangGraph, provide rich ways to build multi-agent solutions to problems. If you're interested in playing with a fun multi-agent system, also check out ChatDev, an open source implementation of a set of agents that run a virtual software company. I encourage you to check out their GitHub repo and perhaps clone the repo and run the system yourself. While it may not always produce what you want, you might be amazed at how well it does. 

Like the design pattern of Planning, I find the output quality of multi-agent collaboration hard to predict, especially when allowing agents to interact freely and providing them with multiple tools. The more mature patterns of Reflection and Tool Use are more reliable. I hope you enjoy playing with these agentic design patterns and that they produce amazing results for you! 

If you're interested in learning more, I recommend: 

“Communicative Agents for Software Development,” Qian et al. (2023) (the ChatDev paper)
“AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation,” Wu et al. (2023) 
“MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework,” Hong et al. (2023)
Keep learning!

Andrew

P.S. Large language models (LLMs) can take gigabytes of memory to store, which limits your ability to run them on consumer hardware. Quantization can reduce model size by 4x or more while maintaining reasonable performance. In our new short course “Quantization Fundamentals,” taught by Hugging Face's Younes Belkada and Marc Sun, you’ll learn how to quantize LLMs and how to use int8 and bfloat16 (Brain Float 16) data types to load and run LLMs using PyTorch and the Hugging Face Transformers library. You’ll also dive into the technical details of linear quantization to map 32-bit floats to 8-bit integers. I hope you’ll check it out! 

News

Custom Agents, Little Coding
Google is empowering developers to build autonomous agents using little or no custom code.

What’s new: Google introduced Vertex AI Agent Builder, a low/no-code toolkit that enables Google’s AI models to run external code and ground their responses in Google search results or custom data.

How it works: Developers on Google’s Vertex AI platform can build agents and integrate them into multiple applications. The service costs $12 per 1,000 queries and can use Google Search for $2 per 1,000 queries.

You can set an agent’s goal in natural language (such as “You are a helpful assistant. Return your responses in markdown format.”) and provide instructions (such as “Greet the user, then ask how you can help them today”). 
Agents can ground their outputs in external resources including information retrieved from Google’s Enterprise Search or BigQuery data warehouse. Agents can generate a confidence score for each grounded response. These scores can drive behaviors such as enabling an agent to decide whether its confidence is high enough to deliver a given response.
Agents can use tools, including a code interpreter that enables agents to run Python scripts. For instance, if a user asks about popular tourist locations, an agent can call a tool that retrieves a list of trending attractions near the user’s location. Developers can define their own tools by providing instructions to call a function, built-in extension, or external API.
The system integrates custom code via the open source library LangChain including the LangGraph extension for building multi-agent workflows. For example, if a user is chatting with a conversational agent and asks to book a flight, the agent can route the request to a subagent designed to book flights.
Behind the news: Vertex AI Agent Builder consolidates agentic features that some of Google’s competitors have rolled out in recent months. For instance, OpenAI’s Assistants API lets developers build agents that respond to custom instructions, retrieve documents (limited by file size), call functions, and access a code interpreter. Anthropic recently launched Claude Tools, which lets developers instruct Claude language models to call customized tools. Microsoft’s Windows Copilot and Copilot Builder can call functions and retrieve information using Bing search and user documents stored via Microsoft Graph.

Why it matters: Making agents practical for commercial use can require grounding, tool use, multi-agent collaboration, and other capabilities. Google’s new tools are a step in this direction, taking advantage of investments in its hardware infrastructure as well as services such as search. As tech analyst Ben Thompson writes, Google’s combination of scale, interlocking businesses, and investment in AI infrastructure makes for a compelling synergy. 
"""

# 2. Initialize Hugging Face encoder
encoder = HuggingFaceEncoder(model="sentence-transformers/all-MiniLM-L6-v2")

# 3. Initialize chunker (StatisticalChunker works well for adaptive chunking)
chunker = StatisticalChunker(encoder=encoder)

# 4. Perform chunking
chunks = chunker(docs=[document])

# 5. Print result
for i, chunk in enumerate(chunks[0]):
    print(f"\n--- Chunk {i + 1} ---\n{chunk}")