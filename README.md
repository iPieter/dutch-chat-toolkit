<h1 align="center">
  <br>
  <a href="pieter.ai/blog/2023/dutch-chat-toolkit/"><img src="https://github.com/iPieter/dutch-chat-toolkit/blob/master/assets/logo.png?raw=true" alt="Logo Dutch RAG-based chat Toolkit" width="200"></a>
  <br>
Dutch RAG-based Chat Toolkit
  <br>
</h1>

<h4 align="center">⚡ Toolkit to create Dutch retrieval-augmented chatbots in 5 minutes ⚡</h4>
<p align="center">
<a href="https://opensource.org/licenses/MIT">
	  <img alt="licence" src="https://img.shields.io/badge/License-MIT-green.svg"/></a>
	  <img alt="python" src="https://img.shields.io/badge/Python-3.10-green.svg?logo=Python&logoColor=white"/>
</p>
<p align="center">
  <a href="https://pieter.ai/blog/2023/dutch-chat-toolkit/">Blog post</a>
</p>
<p align="center">
  <a href="#what-is-this">What is this?</a> •
  <a href="#get-started">Get Started</a> •
  <a href="#contributing">Contribute</a>
</p>



## What is this?
This is a Python CLI toolkit to quickly create a chatbot with a web-based user interface. It has the following features:

- Automatic chunking and embedding (with [RobBERT](https://pieter.ai/robbert)) for document retrieval.
- Scraping configurable URLs for knowledge.
- 8-Bit inference for generation of >15 tokens/sec. 
- Configurable prompts with sensible defaults.
- high-quality generations with low VRAM usage thanks to Mistral-7B.
- Web-UI with Gradio.

## Get Started

### Device requirements
You need a GPU with at least 10.6 GB of VRAM. 

#### Requirements
First you need to activate a virtual environment and install the required dependencies (Pytorch, Huggingface, Gradio, Langchain, ...):

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

After that, the toolkit is ready to run, but there are no documents in the vector store yet. For that you need to run the toolkit (once) with the following flag to scrape the urls mentioned in `sources.txt`.

```bash
python main.py --load-from-scratch
```

This only needs to be done once, unless you update the list of sources. The documents are stored in a vector database at the `--vectors-db` location, the default folder is `vectors/`.

### Running the toolkit
Once the knowledge base is initialized, you can run the server at any time using the following command: 

```bash
python main.py
```

This will open a Gradio http server on that machine, which you can access on the url that is printed in the terminal.

### Creating new topics and domains
The demo of this chatbot toolkit is about the Belgian town Oudenaarde. You can easily change this by updating the topic `--topic Oudenaarde` and updating the list of sources in `sources.txt`. Make sure to run the `--load-from-scratch` command once.

The `--topic` flag will be used as part of a prompt, so keep in mind to make sure it fits the following sentence:

```text
Je bent een expert in {topic}.
```

### Running different models
By default, we use Mistral-7B, but different models are also possible, for instance the Dutch GEITje model:

```bash
python main.py --model-name Rijgersberg/GEITje-7B-chat-v2
```

Note that some models will require a different prompt format. GEITje is a Mistral-7B derivative, so it uses the same `[INST]` tokens. 

### Changing model storage location
The toolkit uses two models, RobBERT and Mistral-7B, which requires ~15 GB of free space. You can change the storage location using the Hugging Face home before running the toolkit.

```bash
export HF_HOME=/your/path
python main.py
```

### Full configuration
The following command illustrates how to change the most important parameters:

```bash
python main.py 
    --load-from-scratch 
    --model-name 'mistralai/Mistral-7B-Instruct-v0.1' 
    --vectors-db vectors/ 
    --chunk-size 1024 # Size of the chunks from the sources
    --title 'OudenaardeGPT' # Shown in the web UI
    --topic 'de Oost-Vlaamse stad Oudenaarde' # Prompt
```

## Contributing
Always welcome to contribute. Just open a pull request or an issue.

In particular the following features are welcome:
- Processing the scraped websites with an LLM to clean HTML artifacts.
- Summarizing the question that a user asks and using that for more accurate retrieval instead of embedding the question.

The demo is about Oudenaarde, but try to keep this part modular so users can change that easily.
