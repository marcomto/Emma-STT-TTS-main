# 🚀 Emma - Bilingual AI Agent with RAG

Emma is a voice-driven AI Agent with RAG that utilizes **Faster-Whisper** for speech recognition and **Piper** for text-to-speech. Powered by **Llama 3.1 (8B)** and an **SQLite** database, Emma can also execute keyboard inputs to write and perform web searches. Web searches can be handled either through Ollama web search (with a valid API Key) or by integrating a local search engine like searxng.


---

## 🛠️ Prerequisites

Before you begin, ensure you have met the following requirements:
* **A microphone**
* **Python** (version = 3.13.7)
* **Ollama** (with llama3.1:8b installed)
* **nomic-embed-text** (llm che fa l'embedding dei messaggi, installa con: ollama run nomic-embed-text)
* **Piper** (with piper voices, default is C:\piper, C:\piper\voices with valid *.onnx e *.json voice files. See the configuration files in config folder for the languages used)
* **searxng** installed on your local machine (if you want the web search)

## 📦 Installation

Follow these steps to set up the development environment locally:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/marcomto/Emma-STT-TTS-main.git
   ```
2. Navigate to the project directory:
   ```bash
   cd EMMA-STT-TTS-MAIN
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

**First of all, read the configuration document here: docs/index.md**

To run the application in development mode, execute the following command:

```bash
python main.py or python main.py --lang it (for italian)
python main.py --lang en (for english)

important: in constants.py you need to configure cuda dlls like this:
CUDA_PATH = C:\Users\[your_user]\AppData\Local\Programs\Python\Python313\Lib\site-packages\nvidia\cublas\bin

or you will get: [ERROR] Library cublas64_12.dll is not found or cannot be loaded
```
voice commands in English:
```
hello (activation word)
command exit (exits the program)
command type (write on keyboard insted of speaking)
command search (search the web **only by keyboard**)

```
voice commands in Italian:

```
ciao (activation word)
comando esci (exits the program)
comando scrivi (write on keyboard insted of speaking)
comando cerca (search the web **only by keyboard**)

---

## 🤝 Contributing & Support

Thank you for your interest in this project! Before opening an issue or a pull request, please keep the following in mind:

* **No Formal Support:** This project is maintained in my spare time. I do not offer any technical support, troubleshooting, or installation assistance.
* **Limited Responses:** I look at issues and pull requests, but I will respond and follow up **only if and when I have the time and capacity to do so**.
* **Pull Requests:** Contributions that fix bugs or improve the project are always welcome, but please note that review times may vary depending on my availability.

Thank you for your understanding and for respecting the time put into open-source development!

---

## 📄 License

This project is licensed under the terms of the **GNU General Public License v3.0 (GPL-3.0)**.

### What this means for anyone using or modifying this code:
* **Attribution**: Anyone using or modifying this software must include the original copyright notice and credit the author.
* **ShareAlike (Copyleft)**: If someone modifies this code or integrates it into a new software, the resulting project **must** be distributed publicly under the same GPL-3.0 license (remaining open source forever). It cannot be included in closed-source proprietary software.

For more details, please see the full license at https://opensource.org/license/gpl-3.0

---

## 👤 Author

* **Marco Mazzetto** - [marcomto]
* Email: marcomto.mm@gmail.com
