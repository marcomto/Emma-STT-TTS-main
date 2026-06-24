Introduction
==============

I apologize to anyone interested in this program, and I apologize in advance. Between work and personal commitments, I have little time to write a comprehensive guide on how the code works, but I think that today, thanks to AI, we can have complete information on how a loop, thread, or queue works in this program. However, I feel it's important to include the configurations that can make life easier for anyone who uses or wants to work with this code. The README.md file explains how to install the program, how to start it, and how to issue voice commands.

Configuration
==============

LANGUAGES
------

The config folder contains respectively **config_en.json** and **config_it.json**. These are the English and Italian language files. If you want to create another language, you'll need to create another JSON file, e.g., config_es.json for Spanish. This file contains language settings, such as:    
- "user_lang", language used by Faster-Whispher  
- "search_lang", language used by searx fro web searching
- "piper_model", "C:\\Piper\\voices\\en_US-amy-medium.onnx", put here the path to the new voice files

The TTS will need the new languages. That is, the C:\Piper folder with the voices subfolder containing the .onnx and .json files.

- "database": "memory.db", the database name. Each language must have its own, ie. memory_es.db.

If not present, the db_manager.py automatically creates the database with all tables needed.

Thee load_config.py file contains the following code:
- parser.add_argument("--lang", choices=["it", "en"], default="it")

If you add a language, you must also put it here.

The application can then be run with python main.py --lang xx, where xx is your language


WEB SEARCH
-----------

The web_search(session, query) function in the ollama_client.py file currently uses searx web search:

    r = session.post(
        "http://localhost:8888/search",
        params={
            "q": query,
            "format": "json",
            "lang": cfg.get("search_lang", "en-US"),
             "categories": "general"
        },
        timeout=20
    ) 
	
if you want to use ollama search, comment out the above code and uncomment 

    # API_KEY = os.getenv("OLLAMA_WEB_SEARCH_KEY")

    # 1) ricerca web Ollama
    """     r = session.post(
            "https://ollama.com/api/web_search",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "max_results": MAX_RESULTS
            },
            timeout=20
        )
		
Note: API_KEY is your API key tied to your ollama account and is required.
Your API key must be placed in the secret.env file at the root of your project and must contain:
	
	OLLAMA_WEB_SEARCH_KEY=xxxxxxxxxx

where xxxxxxxxxx is your own api key

In main.py when the application is launched, searx is also started. To disable it, remove the following functions from main.py:

	start_searxng()

and the following block

    try:
        # closes SearXNG
        stop_searxng()
    except Exception:
        pass

This disables the search engine from starting, if you don't want to use it or if you want to use ollama web search.

CONSTANTS.PY
------------

of interest here:

- LIBRARY = "llama3.1:8b", the LLM model to use
- EMBED_MODEL = "nomic-embed-text", the model that embeds messages and then saves to db
- MAX_RESULTS, the number of results returned by the web search

UTILS.PY
------------------

The interface refers to the console where the program is run.
To make it easier for the user to read, the text will be blue and the wizard will be green.
Error messages will be displayed in red.

class Colors:
    USER = '\033[94m'
    PARTIAL = '\033[93m'
    ASSISTANT = '\033[92m'
    ERROR = '\033[91m'
    RESET = '\033[0m'

MEMORY WORKER
-----------------

This is definitely the most difficult part and deserves to be explained.
The function is essentially an asynchronous consumer with a buffer:

Input: Dialog interactions are placed in the → FACT_QUEUE
Processing: Text transformation → Normalized embedding with the nomic embedder model
Accumulation: Data held temporarily in RAM
Persistence: Writing to SQLite in batches
Synchronization: Clearing the vector cache after each commit
Resilience: If the queue is empty, it waits; if there is an error, it doesn't die but retries.

The key thing to understand is that it doesn't write every item to the DB immediately: it accumulates and flushes periodically to reduce the number of commits.
There are two places where data can live:

Database (memory_vectors)
Cache in RAM (VECTOR_CACHE)

The cache is a temporary copy of data that is used to avoid continuously reading from the database.

             meory request
                    │
                    ▼
             VECTOR_CACHE
                    │
        ┌───────────┴───────────┐
        │                       │
     found                 not found
        │                       │
        ▼                       ▼
uses RAM data         reads from database

Caching speeds up because reading from RAM is much faster than doing a SQLite query.

**The problem**

Suppose VECTOR_CACHE contains:

VECTOR_CACHE = {
    "sessione_123": [
        vettore_A,
        vettore_B,
        vettore_C
    ]
}

Then comes a new fact:

User:
"I like pizza"

The worker does:

text
  ↓
embedding
  ↓
saving in the DB

Now the database contains:

memory_vectors

session_123
 ├── vector_A
 ├── vector_B
 ├── vector_C
 └── vector_D   <-- new

But the cache still contains:

VECTOR_CACHE

session_123
 ├── vector_A
 ├── vector_B
 └── vector_C

So the cache is stale.

If someone searches for information from memory, he might get:

"look for similar memories"

        ↓

VECTOR_CACHE

        ↓

find only A,B,C

        ↓

ignores D

The new memory exists in the database but is not seen.

So what does this part do?
with VECTOR_CACHE_LOCK:
    if session_id in VECTOR_CACHE:
        del VECTOR_CACHE[session_id]

Means:

"I just added new vectors to the database. The copy I have in RAM is no longer reliable. Delete it."

before:

VECTOR_CACHE

session_123:
    A
    B
    C

after:

VECTOR_CACHE

(empty)

To the next request:

search for memory

      ↓

empty cache

      ↓

reload from database

      ↓

A
B
C
D  <-- now there is also the new data
Pwhy is VECTOR_CACHE_LOCK needed?

This part:

with VECTOR_CACHE_LOCK:

it's needed because there are two threads.

Thread 1                 Thread 2
---------                ---------

is reading cache       deletes cache

inconsistent situations could arise.

The lock tells us:

"One thread at a time can modify the cache."

In summary:

"Vector cache clearing" does not delete actual data.

It doesn't do:

DATABASE ❌ clear memory

It does:

CACHE RAM ✅ forget the old copy

It's a technique called cache invalidation:

new data saved
        │
        ▼
old invalid cache
        │
        ▼
deletes cache
        │
        ▼
the next time you access the data, reload the updated data

In the case of the program, this is especially important because it is handling vector embeddings: if the cache is not invalidated, the "memory" system may appear to never learn new information.

NOTE: Although the program uses a message rotation system (maximum 50 messages with SUMMARY_LIMIT = 50), and then summarizes and deletes them, I didn't directly implement a pruning system for the memory_vectors and summaries tables, simply to avoid performing the deletion during program use and therefore to avoid slowing down the audio stream. However, I created a manual script, clean_database.py, which keeps the last 300 records of the memory_vectors table (MAX_VECTORS_PER_SESSION) and the last 30 summaries of the summaries table (MAX_SUMMARIES). However, to avoid overloading the context, that is, the history of conversations that each call carries with it, the program dynamically limits them in the adaptive_memory_tuning() function depending on the conversation flow, whether short, medium, or long.

FASTER WHISPHER AND CUDA
----------------------

The program uses Faster Whispher for audio capture, which takes advantage of the GPU rather than the CPU. CUDA must be installed on the system, and its DLLs must be recognized. I had trouble using CUDA at first, and I solved it by declaring the libraries in the Windows system path:
- C:\Users\\[your_user]\AppData\Local\Programs\Python\Python313\Lib\site-packages\nvidia\cublas\bin

If you don't want/can't use CUDA, you can also use the CPU:

    whisper_model = WhisperModel(
        "small",
        device="cpu",
        compute_type="float16"
    )
