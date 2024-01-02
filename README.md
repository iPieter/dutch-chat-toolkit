<h1 align="center">
  <br>
  <a href="pieter.ai/blog/2023/dutch-chat-toolkit/"><img src="https://raw.githubusercontent.com/context-labs/autodoc/master/assets/logo.png" alt="Logo Dutch RAG-based chat Toolkit" width="200"></a>
  <br>
Dutch RAG-based Chat Toolkit
  <br>
</h1>

<h4 align="center">⚡ Toolkit to create Dutch retrieval-augmented chatbots ⚡</h4>
<p align="center">
<a href="https://opensource.org/licenses/MIT">
	  <img alt="licence" src="https://img.shields.io/badge/License-MIT-green.svg">
      </a>
	  <img alt="python" src="https://img.shields.io/badge/Python-3.10-green.svg?logo=Python&logoColor=white">
</p>
<p align="center">
  <a href="pieter.ai/blog/2023/dutch-chat-toolkit/">Blog post</a>
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
You need a GPU with at least  

#### Requirements

```bash
$ pip install -r requirements.txt
```



### Running the toolkit
Once the knowledge base is initialized, you can run the server at any time using the following command: 

```bash
$ python main.py
```

This will open a Gradio http server on that machine, which you can access following the url in the terminal.

### Creating new topics and domains
The demo of this chatbot toolkit is about the Belgian town Oudenaarde. You can easily change this by updating the topic `--topic Oudenaarde` and

## Contributing